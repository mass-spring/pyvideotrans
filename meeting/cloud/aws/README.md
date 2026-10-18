# AWS EC2（GPU）で声の比較実験を動かす（ec2run）

Google Colab の代わりに、**AWS の GPU パソコン（EC2）** を借りて、`voicelab` の比較実験（V2：現状値の測定、V3：改善策の比較）を動かす道具です。
このクラウド開発環境（Claude Code）から **AWS の API だけ** で操作します。SSH（遠隔ログイン）は使いません。

```
Claude Code（ここ）                          AWS（東京リージョン ap-northeast-1）
  python -m meeting.cloud.aws ...  ──API──▶  EC2 GPU（g4dn.xlarge など）… 外からの接続は一切受けない
        │                                        ▲   │ 実験を動かす（環境づくり → 音声合成 → 測定 → Excel）
        │ ① 入力を S3 に置く ──────▶  S3 バケット（非公開・暗号化・14日で自動削除）
        │ ② SSM で「実行して」と命令 ─────────────┘   │
        └ ③ 結果（Excel・wav）を S3 から取ってくる ◀───┘
              → meeting/outputs/<実験名>/
```

> **データの規則（必ず守る）**
> クラウドに上げてよい音声は、**自分の声／同意を得た模擬会議／公開データ**だけです。
> **会社の会議音声・社外秘の音声は、このAWSアカウントが会社に承認されたものでない限り、上げません。**
> 個人のAWSアカウントは「承認されていない」ものとして扱います（PLAN.md の D4・Q13）。声の複製は本人の同意が必要です。
> 最初の動作確認は `--public-test`（公開データだけ。あなたの声を使わない）で行えます。

---

## 0. 全体の流れ

| 段階 | だれが | 何を | かかる時間 |
|---|---|---|---|
| **A. 準備（1回だけ）** | あなた（AWS コンソール） | 予算アラート → GPU の枠の申請 → 専用 IAM ユーザーと鍵 | 30分 ＋ 枠の承認待ち（数分〜数日） |
| **B. 鍵の登録（1回だけ）** | あなた（Claude Code の環境設定） | `EC2_ACCESS_KEY_ID` など3つ | 5分 |
| **C. 実験（毎回）** | Claude（あなたの了承つき） | `preflight` → `setup` → `launch` → `run` → `teardown` | 1〜3時間（実験の大きさによる） |

C の各段階で、お金がかかる操作（起動・実行）は `--yes` を付けないと動きません。
Claude は `--yes` を付ける前に、**見積と内容をあなたに見せて確認**します。

---

## 1. 準備（AWS コンソールでやること）

> 画面の表示は 2026年10月時点の英語表示です。画面が違うときは、メニュー名を検索してください。
> 1〜2 は **アカウントの持ち主（ルートユーザー、または管理者）** でログインしてやります。3 で作る専用ユーザーでは、請求の画面は見られません。

### 1-1. 予算アラートを先に作る（暴走防止の保険）

1. <https://console.aws.amazon.com/cost-management/> を開く → 左の **Budgets** → **Create budget**
2. **Customize (advanced)** → **Cost budget** → **Next**
3. Budget name: `meeting-translator` / Period: **Monthly** / Budgeting method: **Fixed** / Budgeted amount: **20**（ドル。目安は下の「費用の目安」）
4. **Add an alert threshold**: Threshold `80`、**% of budgeted amount**、**Actual**、メールアドレスを入れる（もう1つ `100%` の **Forecasted** も足すと安心）
5. 届く確認メールのリンクを開いて、メールアドレスを有効にする → **Create budget**

予算アラートは「知らせる」だけで、止めてはくれません。止めるのはこの道具の自動停止（下の「安全のしくみ」）です。

### 1-2. GPU の枠（vCPU のクォータ）を申請する

新しいAWSアカウントは、GPU のインスタンス（G インスタンス）の枠が **0** のことが多く、そのままでは起動できません。

1. 画面右上のリージョンを **Asia Pacific (Tokyo) ap-northeast-1** にする（枠はリージョンごと）
2. <https://ap-northeast-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA> を開く
   （または Service Quotas → AWS services → Amazon EC2 → **Running On-Demand G and VT instances**）
3. **Request increase at account level** → 新しい値に **4**（`g4dn.xlarge` 1台分＝4 vCPU）以上 → **Request**
   - `g5.xlarge` も試すなら 4 のままで足ります（同じく4 vCPU）。2台同時に動かしたいなら 8。
   - **スポット**（`--spot`）も使うなら、**All G and VT Spot Instance Requests**（`L-3819A6DF`）も同様に申請します。
4. 承認はメールで届きます（数分〜数日）。理由を聞かれたら「機械学習（音声合成）の実験用」と書きます。

枠の値は、あとで `preflight` が自動で確かめて、足りなければこの手順を日本語で表示します。

### 1-3. この道具専用の IAM ユーザーを作る

ルートユーザーの鍵は**絶対に使いません**。この道具だけの、権限の小さいユーザーを作ります。

**(a) 権限（ポリシー）を作る**

1. IAM → **Policies** → **Create policy** → **JSON** のタブ
2. 中身を全部消して、下の JSON を貼り付ける（同じ内容が `iam_user_policy.json` にもあります）
3. **Next** → Policy name: `meeting-translator-ec2run` → **Create policy**

- 東京以外のリージョンを使うなら、JSON の `ap-northeast-1` を置き換えます。
- 長さの上限（6,144文字）に収まるよう作ってあるので、「ユーザーに直接書くインライン」ではなく、**この「ポリシーを作る」手順**を使ってください。

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Ec2ReadOnly",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeInstanceTypeOfferings",
        "ec2:DescribeImages",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeVolumes"
      ],
      "Resource": "*"
    },
    {
      "Sid": "Ec2RunInstancesNewResourcesMustBeTagged",
      "Effect": "Allow",
      "Action": "ec2:RunInstances",
      "Resource": [
        "arn:aws:ec2:ap-northeast-1:*:instance/*",
        "arn:aws:ec2:ap-northeast-1:*:volume/*",
        "arn:aws:ec2:ap-northeast-1:*:network-interface/*",
        "arn:aws:ec2:ap-northeast-1:*:spot-instances-request/*"
      ],
      "Condition": {"StringEquals": {"aws:RequestTag/Project": "meeting-translator"}}
    },
    {
      "Sid": "Ec2RunInstancesReferencedResources",
      "Effect": "Allow",
      "Action": "ec2:RunInstances",
      "Resource": [
        "arn:aws:ec2:ap-northeast-1::image/*",
        "arn:aws:ec2:ap-northeast-1:*:subnet/*",
        "arn:aws:ec2:ap-northeast-1:*:security-group/*"
      ]
    },
    {
      "Sid": "Ec2TagOnlyAtCreation",
      "Effect": "Allow",
      "Action": "ec2:CreateTags",
      "Resource": "arn:aws:ec2:ap-northeast-1:*:*/*",
      "Condition": {"StringEquals": {"ec2:CreateAction": ["RunInstances", "CreateSecurityGroup"]}}
    },
    {
      "Sid": "Ec2TerminateOnlyOurInstances",
      "Effect": "Allow",
      "Action": "ec2:TerminateInstances",
      "Resource": "arn:aws:ec2:ap-northeast-1:*:instance/*",
      "Condition": {"StringEquals": {"ec2:ResourceTag/Project": "meeting-translator"}}
    },
    {
      "Sid": "Ec2SecurityGroupCreate",
      "Effect": "Allow",
      "Action": "ec2:CreateSecurityGroup",
      "Resource": [
        "arn:aws:ec2:ap-northeast-1:*:vpc/*",
        "arn:aws:ec2:ap-northeast-1:*:security-group/*"
      ]
    },
    {
      "Sid": "Ec2SecurityGroupOnlyOurs",
      "Effect": "Allow",
      "Action": ["ec2:RevokeSecurityGroupIngress", "ec2:DeleteSecurityGroup"],
      "Resource": "arn:aws:ec2:ap-northeast-1:*:security-group/*",
      "Condition": {"StringEquals": {"ec2:ResourceTag/Project": "meeting-translator"}}
    },
    {
      "Sid": "SsmSendCommandDocument",
      "Effect": "Allow",
      "Action": "ssm:SendCommand",
      "Resource": "arn:aws:ssm:ap-northeast-1::document/AWS-RunShellScript"
    },
    {
      "Sid": "SsmSendCommandOnlyOurInstances",
      "Effect": "Allow",
      "Action": "ssm:SendCommand",
      "Resource": "arn:aws:ec2:ap-northeast-1:*:instance/*",
      "Condition": {"StringEquals": {"ssm:resourceTag/Project": "meeting-translator"}}
    },
    {
      "Sid": "SsmReadCommands",
      "Effect": "Allow",
      "Action": [
        "ssm:GetCommandInvocation",
        "ssm:ListCommandInvocations",
        "ssm:ListCommands",
        "ssm:CancelCommand",
        "ssm:DescribeInstanceInformation"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SsmPublicDlamiParameter",
      "Effect": "Allow",
      "Action": "ssm:GetParameter",
      "Resource": "arn:aws:ssm:ap-northeast-1::parameter/aws/service/deeplearning/ami/*"
    },
    {
      "Sid": "ServiceQuotasReadOnly",
      "Effect": "Allow",
      "Action": [
        "servicequotas:GetServiceQuota",
        "servicequotas:GetAWSDefaultServiceQuota",
        "servicequotas:ListRequestedServiceQuotaChangeHistoryByQuota"
      ],
      "Resource": "*"
    },
    {
      "Sid": "IamRoleAndProfileForThisToolOnly",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole",
        "iam:GetRole",
        "iam:TagRole",
        "iam:PutRolePolicy",
        "iam:DeleteRolePolicy",
        "iam:DeleteRole",
        "iam:CreateInstanceProfile",
        "iam:GetInstanceProfile",
        "iam:TagInstanceProfile",
        "iam:AddRoleToInstanceProfile",
        "iam:RemoveRoleFromInstanceProfile",
        "iam:DeleteInstanceProfile"
      ],
      "Resource": [
        "arn:aws:iam::*:role/meeting-translator-*",
        "arn:aws:iam::*:instance-profile/meeting-translator-*"
      ]
    },
    {
      "Sid": "IamAttachOnlyTheSsmManagedPolicy",
      "Effect": "Allow",
      "Action": ["iam:AttachRolePolicy", "iam:DetachRolePolicy"],
      "Resource": "arn:aws:iam::*:role/meeting-translator-*",
      "Condition": {"ArnEquals": {"iam:PolicyARN": "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"}}
    },
    {
      "Sid": "IamPassRoleToEc2Only",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::*:role/meeting-translator-*",
      "Condition": {"StringEquals": {"iam:PassedToService": "ec2.amazonaws.com"}}
    },
    {
      "Sid": "IamSpotServiceLinkedRoleFirstUse",
      "Effect": "Allow",
      "Action": "iam:CreateServiceLinkedRole",
      "Resource": "arn:aws:iam::*:role/aws-service-role/spot.amazonaws.com/AWSServiceRoleForEC2Spot",
      "Condition": {"StringEquals": {"iam:AWSServiceName": "spot.amazonaws.com"}}
    },
    {
      "Sid": "S3BucketsNamedMeetingTranslator",
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:DeleteBucket",
        "s3:ListBucket",
        "s3:GetBucketLocation",
        "s3:ListBucketMultipartUploads",
        "s3:PutBucketPublicAccessBlock",
        "s3:PutEncryptionConfiguration",
        "s3:PutLifecycleConfiguration",
        "s3:PutBucketTagging",
        "s3:PutBucketPolicy"
      ],
      "Resource": "arn:aws:s3:::meeting-translator-*"
    },
    {
      "Sid": "S3ObjectsInThoseBuckets",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
      "Resource": "arn:aws:s3:::meeting-translator-*/*"
    }
  ]
}
```

この権限でできること／できないこと:

| できる | できない |
|---|---|
| `Project=meeting-translator` のタグを付けたインスタンスの起動・終了（タグが無いものは終了できない） | 他のインスタンスの終了、SSH 用の鍵の作成、外からの接続（インバウンド）の許可 |
| `meeting-translator-` で始まる名前の S3 バケットと IAM ロールの作成・削除 | 他のバケット・他のロールの操作、AWS 全体の管理 |
| SSM でタグ付きインスタンスに命令（`AWS-RunShellScript` だけ） | 請求・ユーザー管理・他のサービス |

注意: このユーザーは `meeting-translator-*` という名前のロールを作れるので、**鍵が漏れたら、その名前のロールを使った悪用はできてしまいます**。鍵は大切に扱い、使わなくなったら削除してください。

**(b) ユーザーを作る**

1. IAM → **Users** → **Create user** → User name: `meeting-translator-ec2run`
2. **Provide user access to the AWS Management Console** は**チェックしない**（コンソールには入れないユーザーにする）
3. **Next** → **Attach policies directly** → さっき作った `meeting-translator-ec2run` を選ぶ → **Create user**

**(c) アクセスキーを作る**

1. 作ったユーザーを開く → **Security credentials** → **Create access key**
2. 用途は **Command Line Interface (CLI)** → 確認にチェック → **Next** → **Create access key**
3. 表示される **Access key ID**（`AKIA...`）と **Secret access key**（長い文字列）を控える。**Secret は、この画面を閉じると二度と見られません。**

> **キーの扱い**: チャットや GitHub、ファイルに貼らないでください。次の「2. 鍵を登録する」の画面にだけ入力します。

---

## 2. 鍵を登録する（Claude Code の環境設定）

Claude Code の**クラウド環境の設定**に、**環境変数**として次の3つを登録します。**値をチャットに貼らないこと。**

| 環境変数の名前 | 入れる値 |
|---|---|
| `EC2_ACCESS_KEY_ID` | 1-3(c) の Access key ID（`AKIA...`） |
| `EC2_SECRET_ACCESS_KEY` | 1-3(c) の Secret access key |
| `EC2_REGION` | `ap-northeast-1`（東京。省略してもこの値になります） |

手順: セッションのタイトルバーにある**クラウド環境のメニュー → Edit（編集）** → 環境変数（Environment variables）に3つを追加 → 保存。
**新しいセッションから有効**です。今のセッションで反映されないときは、新しいセッションを開いてください。

- 名前が `EC2_...` なのは、この環境にもともとある `AWS_ACCESS_KEY_ID`（作業環境が入れた無効な仮の値）とぶつからないようにするためです。この道具は `AWS_*` を**見ません**。
- 一時的な鍵（STS）を使うなら `EC2_SESSION_TOKEN` も登録できます（普通は不要）。
- 鍵はどのコマンドでも画面に表示しません。アカウント ID も下4桁以外は伏せ字（`********1234`）で表示します。

---

## 3. 使い方（順番どおりに）

Claude に「preflight を実行して」のように頼むか、**リポジトリのルート**（`pyvideotrans/`）で次のコマンドを実行します。
`boto3` は `meeting/pyproject.toml` に入れていないので、`--with boto3` を付けます（コピペで動きます）。

```bash
cd /home/user/pyvideotrans
EC2="uv run --no-project --with boto3 python -m meeting.cloud.aws"    # 以降は $EC2 と書きます
```

（`python meeting/cloud/aws/ec2run.py ...` でも同じです。）

### 手順1　preflight：何も作らずに確認（無料・読み取りだけ）

```bash
$EC2 preflight
```

確認すること: 鍵が有効か／アカウント（下4桁）／GPU の枠／`g4dn.xlarge` が東京で提供されているか／起動中のものが無いか／料金の目安／setup が済んでいるか。
最後に `○ 進められます` が出ればOKです。枠が 0 なら、**1-2 の手順を日本語で表示**します（承認待ちの間は進めません）。

### 手順2　setup：土台を作る（1回だけ。S3・IAM・セキュリティグループは無料）

```bash
$EC2 setup            # 作る内容を表示して終わる（何も作らない）
$EC2 setup --yes      # 内容を確認したら、これで作る
```

作るもの（どれも `Project=meeting-translator` のタグ付き。何度実行しても同じ結果になります）:

- S3 バケット（非公開・SSE-S3 暗号化・14日で自動削除・HTTPS 以外は拒否）
- IAM ロール＋インスタンスプロファイル（SSM ＋「そのバケットの中だけ」の S3 権限）
- セキュリティグループ（**外からの接続を1つも許可しない**）

作ったリソースの ID は `meeting/cloud/aws/.state.json` に記録されます（Git に入れません。消えても、決まった名前で探し直せます）。

### 手順3　launch：GPU を起動する（ここからお金がかかる）

```bash
$EC2 launch --type g4dn.xlarge --hours 3          # 見積を表示して終わる（何も起動しない）
$EC2 launch --type g4dn.xlarge --hours 3 --yes    # 見積を確認したら、これで起動
```

- **`--hours` が費用の上限**です。この時間がたつと OS が自動で停止し、インスタンスごと消えます（下の「安全のしくみ」）。
- 見積（上限）は「1時間の料金 × 時間 ＋ ディスク ＋ 公開IPv4」です。`g4dn.xlarge` を3時間で約 **$2.2**（約330円）。
- `--spot` を付けるとスポット（空きがあれば安い。AWS の都合で止まることがある）。
- 起動後、SSM が使えるようになるまで待ち（2〜4分）、**自動停止の予約が本当に入っているか**まで確認して終わります。

### 手順4　run：実験を動かして結果を持ってくる

最初は、**公開データだけ**で通しで動かして、流れを確認します（あなたの声は使いません）。

```bash
$EC2 run --exp exp01 --engines voxcpm2,qwen3tts,chatterbox --public-test --dry-run-first           # 内容を表示して終わる
$EC2 run --exp exp01 --engines voxcpm2,qwen3tts,chatterbox --public-test --dry-run-first --yes     # 実行
```

自分の声（登録済みの `profiles/A`）で測るとき:

```bash
$EC2 run --exp v2_A --engines voxcpm2,qwen3tts,chatterbox --speaker A=meeting/profiles/A --dry-run-first --yes
```

- `--speaker 名前=フォルダ`：`voiceprofile` が作ったフォルダ（`profile.json` がある）か、wav を入れたフォルダ。複数指定できます。
- 設定ファイルを使うなら `--config exp.json`（`voicelab` の設定。パスは自動でインスタンス用に書き換えます）。文を変えるなら `--targets my_targets.csv`。
- `--mode quick`（お試し。既定）／`--mode full`（繰り返し2回・全部の文）。
- `--dry-run-first`：最初にダミー音声で「生成 → 測定 → Excel」の流れを確かめ、間違いがあればGPUの時間を使う前に止めます（おすすめ）。
- `--engines cosyvoice3` も指定できますが、導入が一番壊れやすいです（最初は外す）。
- やっていること（ノートブックと同じ）:
  エンジンごとに**別の仮想環境**（uv）を作る → エンジンごとに `generate` → 最後に `score`（測定）→ Excel。
  ライブラリのバージョンは `voice_baseline_colab.ipynb` の固定と同じです（テストで一致を確認しています）。
- 実行中は、インスタンスのログが `  | ...` の形で流れます。切れても実験は続きます。**つなぎ直す**:
  ```bash
  $EC2 run --exp exp01 --attach
  ```
- 結果は `meeting/outputs/<実験名>/` に入ります（Git に入れません）。**見るのは `voicelab_result.xlsx`**。読み方は `meeting/src/voicelab/README.md` の6章です。
  wav が不要なら `--skip-audio`。あとから取るだけなら（S3 には14日残ります）:
  ```bash
  $EC2 fetch --exp exp01
  ```
- 同じ実験名の結果が S3 に残っているとき: 続きから作る `--resume`／最初から作り直す `--fresh`／別の名前にする。
- GitHub に push していない変更や、非公開リポジトリのときは `--code local`（この環境の `meeting/src` をアップロード）。
- 終わったら自動でインスタンスを終了したいときは `--auto-terminate`（1分後に停止＝終了。結果は先に S3 に保存されます）。

> Claude Code で長い実行を頼むときは「`run` をバックグラウンドで実行して、ときどきログを見て」と伝えてください（1回のコマンドの時間制限を超えるため）。

### 手順5　teardown：止める（使い終わったら必ず）

```bash
$EC2 status                 # 動いているもの・経過時間・ここまでの費用の目安
$EC2 teardown               # 終了させるものを表示して終わる
$EC2 teardown --yes         # 終了。terminated になるまで確認して、残っているもの（ディスク・バケット）の費用も報告
```

- 終了させるのは、**タグ（`Project=meeting-translator` かつ `ManagedBy=ec2run`）が付いていて、状態ファイルに記録のあるインスタンス**だけです。記録が無いが同じタグのものも終わらせるなら `--all`。タグが無いインスタンスは、`--all` でも触りません。
- S3 バケットを今すぐ消す（結果を回収したあと）: `$EC2 teardown --delete-bucket --confirm-bucket <バケット名> --yes`（バケット名は `status` に出ます）
- IAM ロール・インスタンスプロファイル・セキュリティグループも消す: `--delete-setup`（無料なので、残してもかまいません）

---

## 4. 費用の目安

東京リージョン・Linux・オンデマンドの1時間あたりの料金（2026-10-15 の AWS 公開価格表から。**概算（approximate, check AWS pricing）です。必ず AWS の料金ページで確認してください**）。

| タイプ | vCPU | GPU | $/時間 | 1時間 | 3時間 | 6時間 |
|---|---|---|---|---|---|---|
| **g4dn.xlarge**（既定） | 4 | T4 16GB | 0.71 | $0.73 | **$2.18** | $4.37 |
| g5.xlarge | 4 | A10G 24GB | 1.459 | $1.48 | $4.43 | $8.86 |
| g4dn.2xlarge | 8 | T4 16GB | 1.015 | $1.03 | $3.10 | $6.20 |
| g5.2xlarge | 8 | A10G 24GB | 1.758 | $1.78 | $5.33 | $10.66 |
| g6.xlarge | 4 | L4 24GB | 1.167 | $1.19 | $3.56 | $7.11 |

- 上の「N時間」は、**本体 ＋ ディスク（gp3 100GB ≒ $0.013/時間）＋ 公開IPv4（≒ $0.005/時間、AWSの発表値 (要確認)）** を足した上限です。為替・消費税・データ転送は含みません。
- 時間の目安（**推定・未実測**。最初の実測で書き換えます）: 環境づくり 20〜40分（エンジンごと。初回のみ）、生成 `quick` で 10〜20分、測定 5〜10分 → **`quick` で 1.5〜2.5時間 ≒ $1.1〜$1.8＋α、`full` で 3〜4時間**。
- S3 は保管が数十 MB なら1円未満（14日で自動削除）。S3 や EBS を残すと少額がかかり続けます（`teardown` が報告します）。
- 月 $20 の予算アラートを置いておけば、`quick` を十数回回しても収まります。

---

## 5. 安全のしくみ

費用が膨らまないよう、何重にも歯止めをかけています。

1. **`--yes` が無いと、作らない・起動しない・消さない**（`setup` / `launch` / `run` / `teardown`）。何をするかを先に表示します。
2. **自動停止（ハードな上限）**: `launch --hours N` は、起動時に `shutdown -h +<N×60分>` を予約し、**停止＝終了（`InstanceInitiatedShutdownBehavior=terminate`）** にします。起動後に SSM で予約を確かめ、無ければ入れ直します。
   - 予約は OS を再起動すると消える可能性があります（未確認）。`status` が「予定時刻を過ぎて動いている」ものを警告します。
   - 時間が足りないときは、途中結果が S3 に残るので、新しく `launch` して `run --resume` で続きから作れます。
3. **二重起動の防止**: すでに起動中のものがあると、`launch` は断ります（`--allow-multiple` で上書き）。
4. **外から入れない**: セキュリティグループのインバウンドは 0 件、IMDSv2 必須、SSH 用の鍵なし。操作は SSM（AWS の API）だけです。
5. **S3**: 非公開・暗号化・14日で自動削除・HTTPS 以外は拒否。インスタンスのロールは「そのバケットの中」しか触れません。
6. **タグ**: この道具が作ったものすべてに `Project=meeting-translator` / `ManagedBy=ec2run`。`teardown` と `status` はこのタグのものしか触りません（終了の直前にも、タグを確かめ直します）。
7. **鍵を表示しない**: 鍵は環境変数から読み、どの出力にも出しません。エラーに紛れても伏せ字にします。アカウント ID も下4桁だけ。
8. **TLS の検証は切らない**: この環境のプロキシの CA 証明書（`/root/.ccr/ca-bundle.crt`）を使います。

---

## 6. うまくいかないとき

| 症状 | 対処 |
|---|---|
| `AWS の鍵が環境変数に設定されていません` | 「2. 鍵を登録する」。登録したのに出るなら、**新しいセッション**を開く |
| `鍵が無効です` | キーを貼り間違えた／ユーザーのキーを無効化した。新しいキーを作って登録し直す（古いキーは削除） |
| `権限が足りません` | 1-3(a) のポリシーが、このユーザーに付いているか確認。ポリシーを更新したら貼り直す |
| `preflight` が「不足」「0 vCPU」 | 1-2 の枠の申請。「申請中」と出たら承認メールを待つ |
| `VcpuLimitExceeded` | 同上（今の枠 < 起動に必要な vCPU）。他の GPU インスタンスが動いていないかも確認 |
| `InsufficientInstanceCapacity` | その時間は AWS に空きがない。別の AZ は自動で試します。時間をおく／`--type g5.xlarge`／`--spot` を外す |
| `SSM が Online になりませんでした` | プロファイルの反映待ち（数分）が多い。`status` で確認して少し待つ。ずっと駄目なら `teardown --yes` して `launch` からやり直す |
| `GPU が見つかりません` | GPU 付きのタイプか確認（`g4dn` / `g5` / `g6`）。AMI は「Deep Learning Base OSS Nvidia Driver GPU AMI」です |
| `リポジトリを取得できませんでした` | 非公開リポジトリか、まだ push していない。`--code local` を付ける |
| `ダミーでの確認に失敗しました` | 設定か声のデータの間違い。`ec2run_logs/run.log` の上のほうに `voicelab` のエラーがあります |
| 途中で自動停止が来た | `status` で確認 → 新しく `launch` → `run --exp 同じ名前 --resume`（できている音声は作り直しません） |
| ログが流れない | しばらく（環境づくり中）は出ません。`status` と、画面の `[段階]` 表示を見る |
| 全部を一度にやり直したい | `teardown --yes`、必要なら `--delete-bucket` → `setup --yes` から |

詳しいエラーを見るには、環境変数 `EC2RUN_DEBUG=1` を付けて実行します。

---

## 7. 確かめられていないこと（本物の AWS でまだ動かしていません）

この道具は、AWS の鍵の無い環境で作りました。次は**偽の AWS（moto）と偽の部品でのテスト**までです（`pytest` 200件以上。下の「開発者向け」）。本物の AWS・GPU で初めて分かることがあります。

- AWS の API の呼び方は公式ドキュメントで確認しましたが、**実際に通るかは初回の `preflight` → `setup` → `launch` で分かります**。IAM ポリシーも、実環境の IAM では未検証です（足りない権限があれば `権限が足りません` と出ます）。
- DLAMI の公開 SSM パラメータ名・クォータコードは公式ページで確認済み（下の出典）。**AMI の中身**（Python 3.10・boto3・SSM エージェントが入っていること）は、リリースノートの記載からの推定です。
- 4つのエンジンの導入と実際の音声合成、T4 での速度とメモリ、CosyVoice3 の導入、モデルのダウンロード時間は、**GPU インスタンスでの初回実行で分かります**（`voicelab/README.md` の「実 GPU でしか確認できないこと」と同じ）。
- 自動停止の予約が再起動後も有効かは確認していません。予約が入っているかの確認（`/run/systemd/shutdown/scheduled`）も、実機での初回確認が必要です。
- 費用は概算です。スポット料金は変動します。

最初の実機実行のあと、実測した時間と費用を PLAN.md の作業ログと、`voicelab/README.md` の「時間と GPU の目安」に書き戻してください。

---

## 8. 開発者向け

| ファイル | 役割 |
|---|---|
| `cli.py` / `ec2run.py` / `__main__.py` | 入口（`python -m meeting.cloud.aws ...` ／ `python meeting/cloud/aws/ec2run.py ...`） |
| `common.py` | 鍵（環境変数）、boto3 の接続（遅延 import）、状態ファイル、料金表、タグ、伏せ字 |
| `preflight.py` / `provision.py` / `launch.py` / `runner.py` / `teardown.py` | 各サブコマンド |
| `ssmrun.py` | SSM Run Command の補助、S3 ログの追跡 |
| `recipes.py` | エンジンごとの環境の作り方（ノートブック 5-0〜5-4 と同じ固定） |
| `remote_job.py` / `prepare_public.py` | **インスタンスの上で動く**部分（標準ライブラリだけ。S3 経由で配る） |
| `iam_user_policy.json` | ユーザーに付ける最小の権限（上の JSON と同じ。テストで一致を確認） |
| `.gitignore` | `.state.json`（リソース ID を含む）を Git に入れない |

テスト（CPU・通信なし。moto は S3/IAM/EC2/STS、SSM と Service Quotas は小さな偽物）:

```bash
cd /home/user/pyvideotrans/meeting
uv run --with boto3 --with moto pytest -q src/tests/test_cloud_aws_*.py
```

- ポリシーのテストは、全コマンドを偽の AWS 上で流して**呼ばれた AWS の操作をすべて記録**し、`iam_user_policy.json` で許可されているかを確認します（権限の過不足の早期発見）。
- `meeting/pyproject.toml` / `uv.lock` には何も足していません（boto3 は `uv run --with boto3` で都度）。

確認した AWS の仕様（出典）:

- DLAMI の最新 AMI を返す公開 SSM パラメータ（Ubuntu 22.04）: `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id`（24.04 は末尾の版を `24.04` に）
  <https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-22-04.html> ／ <https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html> ／ 形式の説明 <https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html>
  （対応: G4dn・G5・G6 など。SSM エージェント入り。`ssm:GetParameter` が必要）
- クォータ: `L-DB2E81BA`＝Running On-Demand G and VT instances、`L-3819A6DF`＝All G and VT Spot Instance Requests（既定 0、vCPU 単位）
  <https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html> ／ <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html>
- SSM の出力: `GetCommandInvocation` の標準出力は先頭 24,000 文字・標準エラーは先頭 8,000 文字で、実行が終わるまで空。`SendCommand` の文書は 64KB まで。→ 大きな中身は S3 に置き、ログは S3 から読む
  <https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_GetCommandInvocation.html> ／ <https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_SendCommand.html>
- `SendCommand` のタグによる絞り込み: <https://docs.aws.amazon.com/systems-manager/latest/userguide/run-command-setting-up.html>
- Service Quotas の API（`GetServiceQuota` は適用値が無いと取れないことがある → `GetAWSDefaultServiceQuota` に切り替え）: <https://docs.aws.amazon.com/servicequotas/2019-06-24/apireference/API_GetServiceQuota.html>
- 料金: AWS Price List（`AmazonEC2`・`ap-northeast-1`、2026-10-15 版）の Linux・共有テナント・オンデマンド。EBS gp3 は $0.096/GB-月

仕様を変えたら、`meeting/docs/PLAN.md` の「6. 決定事項」「9. 作業ログ」「10. 変更履歴」も更新してください（CLAUDE.md の規則）。
