# A19：社交媒体用户画像生成

本项目为 Python 机器学习课程作业，主题为“社交媒体用户画像生成”。项目包含模拟社交媒体用户行为数据的机器学习主流程，以及真实微博数据扩展分析。

## 一、项目结构

```text
A19_社交媒体用户画像生成/
├── data/
│   ├── raw_social_media_users.csv
│   ├── cleaned_social_media_users.csv
│   └── real_weibo/
├── outputs/
│   ├── clustered_social_media_users.csv
│   ├── cluster_summary.csv
│   ├── dbscan_cluster_summary.csv
│   ├── eda_summary.csv
│   ├── k_selection_scores.csv
│   ├── training_log.txt
│   ├── figures/
│   ├── model/
│   └── real_weibo/
├── src/
│   ├── 01_generate_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda_visualization.py
│   ├── 04_train_cluster_model.py
│   ├── 05_generate_report_tables.py
│   ├── 06_real_weibo_analysis.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## 二、环境依赖

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

依赖库包括 `numpy`、`pandas`、`matplotlib`、`seaborn`、`scikit-learn`。

## 三、模拟数据生成逻辑

模拟数据生成脚本为 `src/01_generate_data.py`。

本项目不使用“预设四类用户再生成数据”的方式，而是采用行为特征驱动的生成逻辑：依据社交媒体用户画像研究中常见的行为维度，构造多个连续潜在行为因子，再由这些因子共同影响最终字段。

潜在行为因子包括活跃度、互动倾向、内容创作、夜间活跃、视频消费、商业兴趣和社交规模。聚类标签不是生成阶段写入的数据，而是在后续 K-Means 聚类后，根据各簇行为特征均值进行解释和命名。

## 四、模拟数据主流程

按顺序运行：

```bash
python src/01_generate_data.py
python src/02_clean_data.py
python src/03_eda_visualization.py
python src/04_train_cluster_model.py
python src/05_generate_report_tables.py
```

脚本说明：

- `01_generate_data.py`：基于潜在行为因子生成不少于 1000 条模拟用户行为数据，并加入少量缺失值、异常值和重复值。
- `02_clean_data.py`：完成缺失值处理、异常值处理、统一格式和去重。
- `03_eda_visualization.py`：生成描述性统计表、分布图、相关性热力图和互动散点图。
- `04_train_cluster_model.py`：完成标准化、K=2 到 K=8 的 silhouette 对比实验、固定 K=4 的 K-Means 聚类、PCA 降维可视化和 DBSCAN 对比实验。
- `05_generate_report_tables.py`：生成聚类结果表、用户画像汇总表和群体对比图。

K 值说明：

K=2 在 Silhouette Score 上最高，但本文结合业务解释性选择 K=4 作为最终用户画像数量。K=4 的目的不是追求最高轮廓系数，而是为了形成更适合作业展示和用户画像解释的四类群体。

主要输出：

- `data/raw_social_media_users.csv`
- `data/cleaned_social_media_users.csv`
- `outputs/clustered_social_media_users.csv`
- `outputs/cluster_summary.csv`
- `outputs/dbscan_cluster_summary.csv`
- `outputs/eda_summary.csv`
- `outputs/k_selection_scores.csv`
- `outputs/training_log.txt`
- `outputs/model/kmeans_model.pkl`
- `outputs/figures/distribution.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/engagement_scatter.png`
- `outputs/figures/k_selection.png`
- `outputs/figures/pca_clusters.png`
- `outputs/figures/cluster_profile.png`

最终画像标签：

- `Night Owl`
- `Interactive User`
- `Content Creator`
- `Low-frequency Browser`

## 五、真实微博数据扩展分析

真实微博数据扩展脚本为 `src/06_real_weibo_analysis.py`。运行：

```bash
python src/06_real_weibo_analysis.py
```

扩展模块完成：

- 人口属性分析：清洗性别、出生年、省份、城市字段，计算年龄，并生成性别、年龄、省份分布图。
- 社交关系分析：解析微博关注关系，统计出度、入度、总度数，并生成网络度分布图。
- 汇总输出：合并用户属性与网络特征，输出 CSV 和文字报告。

扩展模块输出：

- `outputs/real_weibo/real_user_profile.csv`
- `outputs/real_weibo/real_network_features.csv`
- `outputs/real_weibo/real_weibo_summary.csv`
- `outputs/real_weibo/real_weibo_report.txt`
- `outputs/real_weibo/figures/gender_distribution.png`
- `outputs/real_weibo/figures/age_distribution.png`
- `outputs/real_weibo/figures/province_top10.png`
- `outputs/real_weibo/figures/network_degree_distribution.png`

## 六、答辩说明建议

本项目中的模拟数据用于展示机器学习建模流程。数据生成没有直接写入最终用户画像类别，而是根据文献和业务调研中常见的社交媒体行为维度，构造连续行为倾向，再生成用户行为特征。K-Means 聚类负责从这些特征中发现用户群体结构，画像名称是对聚类结果的解释。

真实微博数据扩展分析用于补充现实数据场景下的人口属性和社交网络关系统计，使项目既有可控的模拟建模流程，也有真实数据分析支撑。
