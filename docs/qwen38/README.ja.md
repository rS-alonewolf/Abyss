# Qwen3.8-Flash-Next を Abyss で動かす

English: [README.md](README.md)

Qwen3.8-Flash-Next（llama.cpp の `qwen4exp`）は、v1.3.74 から Abyss の速い経路（`--glm-pack`）で動きます。プロンプト
経路、検証窓、MTP ドラフターも使えます。古いエンジン経路（[docs/maya/ORCA.md](../maya/ORCA.md)）もそのまま動きますが、
速い経路のほうがプロンプトは数倍、答えも速くなります。

計測は orcarouter の
[Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) IQ3_XXS
（2 シャード、85.2 GB）で、AI PRO R9700（32 GB、PCIe Gen4 x4）、RAM 94 GB（`STRATA_GLM_RAM_GB=60`）、iGPU レーンありの
環境です。NVIDIA ではこの経路をまだビルドも計測もしていません。

## モデルの形

- **48 層**: delta-net 層が 36（出力ゲートは sigmoid）、4 層ごとに注意層が 12（クエリ 24 ヘッド × 256、KV 2 ヘッド、
  q / k の norm、RoPE 64 次元、sigmoid の出力ゲート）。
- **残差は 4 本の流れ**（各 2560）で、低ランクの hyper-connection が混ぜます（norm のテンソルは無く、混ぜる処理が norm を
  兼ねます）。
- **層 1 の per-layer embedding（PLE）**: 直前のトークンの n-gram のハッシュで、28.8 GB の表から 16 行を引きます。表は SSD
  に置いたまま直接読みます（1 トークン約 1 ms）。
- **QSA**（注意層）: 2,051 セルを超えると、索引が選んだ 4 セル × 512 ブロックだけに注意します（2,051 セルまでは全部に注意）。
- **1 層 512 エキスパート、10 個を選択**、加えて sigmoid ゲート付きの共有エキスパート。幅はどれも 640。
- GGUF には MTP（下書き）ブロックが無いので、ドラフターは別ファイルです（下で説明）。

## 準備

1. **pack**（密な重みを GGUF のままの形式で。BF16 への変換はしません）:

   ```sh
   .venv/bin/python tools/iq_pack.py --fast \
     --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
     --out /path/to/IQ3_XXS/pack-fast
   ```

   2 つのシャードがどちらもダウンロードし終わっている必要があります。pack は約 300 MB で、エキスパートと PLE の表は GGUF
   から直接読みます。

2. **ドラフター（任意、おすすめ）**: 古い経路と同じ MTP の実行用ファイルを、同じ手順で作ります
   （[docs/maya/ORCA.md](../maya/ORCA.md#preparation): `tools/mtp_fetch.py`、`tools/mtp_pack.py --experts q2_0`、
   `tools/mtp_rt.py`、最後に `data/draft_vocab.bin` をコピー）。できたディレクトリには `dense.bin`、`dense.txt`、
   `experts.bin`、`draft_vocab.bin`（計約 820 MB）が入ります。エンジンはこれを VRAM に置きます（キャッシュ込みで約 1 GB）。

3. **設定ファイル**（`abyss-<名前>.json`。[docs/SETTINGS.ja.md](../SETTINGS.ja.md) を参照）:

   ```json
   {
    "exe": "build-hip/strata",
    "args": ["--glm-pack", "/path/to/IQ3_XXS/pack-fast", "--max-context", "32768", "--prefill", "auto"],
    "tokenizer": "/path/to/IQ3_XXS/pack-fast/tokenizer",
    "model_name": "qwen3.8-flash-next-uncensored-iq3_xxs",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "port": 8082,
    "backend": "hip",
    "env": {
     "STRATA_GLM_RAM_GB": "60",
     "STRATA_Q4X_MTP": "/path/to/mtp/rt"
    }
   }
   ```

   起動は `.venv/bin/python serve/server.py --engine strata --config abyss-<名前>.json --port 8082` です（実際に使われる
   ポートはコマンドラインの `--port` です）。

## 設定項目

| 変数 | 既定 | 内容 |
| --- | --- | --- |
| `STRATA_Q4X_MTP=<ディレクトリ>` | なし | ドラフターの実行用ファイル。無いときは、繰り返し（n-gram）からだけ下書きを作ります |
| `STRATA_Q35_MTP_K` | 4（このモデル） | 1 回に作る下書きの最大数（1〜5） |
| `STRATA_Q35_MTP_MIN` / `STRATA_DS4_DSPARK_THR` | 0.75（このモデル） | この確信度以上の下書きだけを検証窓に入れます |
| `STRATA_Q4X_MTP_FULLHEAD=1` | オフ | ドラフターの head を、`draft_vocab.bin` の 10.6 万トークンではなく全トークンにします |
| `STRATA_Q4X_DENSE=1` | オフ | デバッグ用: QSA の選択を使わず、すべてのセルに注意します |
| `STRATA_Q4X_PLE_SYNC=1` | オフ | デバッグ用: 検証窓の PLE の行を、層 0 と並行ではなく層 0 の前に読みます |
| `STRATA_GLM_NO_MTP=1` | オフ | `STRATA_Q4X_MTP` があってもドラフターを切ります |

## 計測（R9700、2026-10-28）

| | 古い経路（v1.3.73） | 速い経路（v1.3.74） |
| --- | ---: | ---: |
| プロンプト 30K トークン | 217 tok/s | 1,037 tok/s |
| 答え、文章（日本語 / 英語） | 52 tok/s | 64 / 65 tok/s |
| 答え、コードを書く | 66〜85 tok/s | 86 tok/s |
| 答え、コードのコピー | | 100〜107 tok/s |
| 答え、ドラフターなし | | 51〜53 tok/s |
| 31K トークンのプロンプトの合言葉 | 正解 | 正解 |
| 会話の次のターン（10.8K トークン） | 再利用 | 再利用: 10,802 トークン、0.28 秒 |

文章とコードの数字は、コマンドラインで貪欲生成した 300 トークンの答えのものです。サーバーの答えは、iGPU レーンや
エキスパートの置き場所の違いで回ごとに少し変わります。下書きの採用率は文章で 74%、コードで 86% です。検証窓 1 回は
30 ms、1 トークンだけなら 18.6 ms です。

品質: 5,600 トークンより後の 494 位置で llama.cpp と比べると、トークン経路で KL 0.026 / top-1 92.3%、プロンプト経路で
0.035 / 90.9% でした。このモデルでは llama.cpp 自身の HIP と CPU の差が KL 0.045 です。貪欲生成では、下書きがあっても
無くても同じトークンが出ます。

## 制限

- GPU は 1 枚: 複数枚への分割はこのモデルでは試していません。ドラフターは 1 枚でモデル全体を動かすときだけ読み込まれます。
- `--uncensor` の編集ファイルはこのモデル用にはありません（ここで使った GGUF はもともと無検閲版です）。
- この経路の NVIDIA でのビルドは確認していません。
