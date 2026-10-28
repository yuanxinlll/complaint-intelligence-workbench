# Complaint Intelligence Workbench

用于 Solix AI Hackathon 2026 Data + AI 方向的本地投诉分类学习原型。
把公开投诉正文转换为问题类别、模型分数和人工复核提示。阶段 0–4 已完成，已有 Streamlit 本地交互界面与从零讲解页。
只使用 pandas、scikit-learn、joblib 等本地 Python 工具，不需要 API key，不向第三方模型上传正文。

## 当前真实结果

CFPB 2024 年 11–12 月 Debt collection：11,706 条有正文标签记录，去除规范化重复后 9,177 条。
训练 6,351，验证 1,428，独立测试 1,398。只用正文；保留全部 7 类。

| 模型 | 验证 Accuracy | 验证 Macro-F1 | 测试 Accuracy | 测试 Macro-F1 |
|---|---:|---:|---:|---:|
| 最多类别基线 | 44.47% | 0.0879 | 46.57% | 0.0908 |
| TF-IDF + LogisticRegression | 52.59% | 0.3975 | 48.50% | 0.3888 |

验证集选择并冻结阈值 0.45：验证未标记部分 368 条，准确率 81.52%；测试未标记部分 325 条（23.25%），准确率仅 70.77%，其中仍有 95 条错误。
这不是可靠的自动处理方案。needs_review=false 仅表示分数达到实验门槛，不是质量保证。
模型分数未经校准，低分也不能可靠检测未知类别。完整汇总见 reports/SHARED_RESULTS.md。

## 打开操作窗口和从零讲解

本地模型和规则准备好后，在项目根目录运行：

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

打开 http://127.0.0.1:8501 操作分类；打开 http://127.0.0.1:8501/?view=learn 阅读最简单的完整讲解。
点击分类只读取现有模型，不会重新训练。页面支持单条分类、CSV 批量分类、筛选、类别分布和当前筛选结果导出。
CSV 必需 narrative 列，可选 complaint_id、issue；原始 issue 显示为 original_issue，不作为预测特征。
仅 UTF-8；最多 5 MB、500 行，单条最多 50,000 字符、总正文最多 200 万字符。异常行保留并标明错误；完全重复正文保留并提醒，因此批次统计也包含重复行。
演示按钮的 3 条文本是人为编写，只用于体验操作。没有本地模型时页面给出说明；新克隆需先按以下步骤生成模型。

## 同学如何运行

建议 Python 3.14；本次运行的 Python 为 3.14.6，依赖版本在 requirements.txt。
在自己的项目副本中建立虚拟环境，安装本地依赖（此命令需要联网下载包）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-evaluation.txt
python -m unittest discover -s tests -v
```

Windows 激活命令为 `.venv\Scripts\activate`。
测试使用明确标记的合成夹具，不需要下载投诉数据，不代表真实模型评估。

### 1. 手动获取官方历史数据

当前 CFPB API 已移除正文，不要用当前 API 的结构化记录替代正文训练。
从 [CFPB 官方历史正文档案](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)下载 “CCDB Export November 2024 through December 2024”。
本次用户提供原文件为约 348 MB；未随仓库上传。实际文件是否相同可核对 SHA-256：
`2ef7d563bf6283a1282bcfba2d99d72f9bfa4a4617524682de72a9b3fa47bed0`。

### 2. 检查并生成小规模子集

在仓库根目录运行，把文件路径替换为自己电脑的路径：

```bash
python src/inspect_archive.py /path/to/downloaded_archive.csv
```

保留原始文件不修改；脚本按产品、非空正文和标签筛选，不排除问题类别。
子集保存到 data/raw/debt_collection_2024_nov_dec.csv，约 14.6 MB。

### 3. 训练并查看过程

```bash
python -u -m src.train
```

终端显示六步进度，日志保存在 artifacts/training.log。另一个终端可运行：

```bash
tail -f artifacts/training.log
```

脚本保留最早的规范化相同正文，包括标签冲突组；不使用后续标签改变最早标签。
固定日期划分：训练 11/1–12/12、验证 12/13–12/21、测试 12/22–12/31。
TF-IDF 只在训练集拟合，不将 Issue、Sub-issue 或响应结果作为输入。

### 4. 冻结复核规则，再独立评估

```bash
python -u -m src.evaluate
```

仅验证集选择规则：网格门槛 0.15–0.975，步长 0.025；未标记样本至少 100 条且观察准确率至少 80%，优先覆盖最多。没有合适门槛则全部建议复核。
规则落盘冻结后才评估测试集；不再根据测试结果调整它。
本地 artifacts 中保存各类指标、混淆矩阵、错误案例和带原文的预测结果；reports 中保存图表。
作图需要 matplotlib；如需要重新运行评估，可安装 requirements-evaluation.txt 中记录的版本。

### 5. 单条分类

```bash
python -m src.predict --text 'I asked for written proof of this debt but received no notice.'
```

此句是人为编写的演示输入。输出 predicted_issue、model_score 和 needs_review；没有匹配规则时 needs_review 为 null。
输入正文为空或超过 50,000 字符会给出错误。只读取自己训练或可信来源的模型文件。

脚本拒绝覆盖原始子集和已有训练/评估运行；要独立重现，请使用没有本地输出文件的新克隆，保留已有运行。

## 协作方式

仓库仅共享代码、依赖、测试、文档和汇总图，不共享原始 CSV、处理后的投诉、模型、含正文的错误案例或本机绝对路径元信息。
每位同学独立从官方获取数据、在本地训练。新克隆里没有模型是正常情况。
推荐每人开一个分支，修改后提交 Pull Request，再由队友审阅合并。
不要提交密码、token、.env 或未经约定的数据。根目录 .gitignore 已排除本地输出。
本项目目前没有独立授权许可证；不要假设公共再分发权限。

## 能力和限制

单标签分类不能覆盖投诉全部诉求。标签由消费者选择，可能有噪声和重叠；两个月公开数据不是代表性抽样，类别不均衡，近似模板仍可能存在。
不用于金融建议、信用判断、真实欺诈识别、公司质量排名、退款决定或根因诊断。
未接入 Solix 产品，不声称获得官方支持。没有部署或自动提交比赛。
下一阶段为整体验证和演示准备，需用户确认后再开始。
