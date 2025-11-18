"""
关卡特征对比分析工具
根据流量好的关卡列表，对比分析流量好和流量不好的关卡特征差异
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class LevelFeatureAnalyzer:
    """关卡特征分析器"""
    
    def __init__(self, features_file: str = "level_features.xlsx"):
        """
        初始化分析器
        
        Args:
            features_file: 特征文件路径（Excel格式）
        """
        self.features_file = Path(features_file)
        self.df = None
        self.good_levels = None
        self.bad_levels = None
        
    def load_features(self):
        """加载特征数据"""
        if not self.features_file.exists():
            raise FileNotFoundError(f"特征文件不存在: {self.features_file}")
        
        self.df = pd.read_excel(self.features_file)
        print(f"成功加载特征数据: {len(self.df)} 个关卡")
        return self.df
    
    def classify_levels(self, good_levels_list: List[str]):
        """
        根据流量好的关卡列表分类关卡
        
        Args:
            good_levels_list: 流量好的关卡ID列表
        """
        if self.df is None:
            raise ValueError("请先加载特征数据")
        
        print(f"\n加载流量好的关卡列表: {len(good_levels_list)} 个")
        
        # 标准化关卡名称（去除level前缀，统一处理）
        def normalize_level_name(name):
            name_str = str(name)
            # 去除level前缀
            if name_str.lower().startswith('level'):
                name_str = name_str[5:]
            return name_str
        
        # 标准化好的关卡列表
        good_levels_normalized = [normalize_level_name(level) for level in good_levels_list]
        
        # 调试信息：显示匹配过程
        print(f"\n调试信息:")
        print(f"输入的流量好关卡列表: {good_levels_list}")
        print(f"标准化后的流量好关卡列表: {good_levels_normalized}")
        print(f"特征文件中的关卡名称: {self.df['关卡名称'].tolist()}")
        
        # 在DataFrame中添加分类列
        self.df['流量分类'] = self.df['关卡名称'].apply(
            lambda x: '好' if normalize_level_name(x) in good_levels_normalized else '不好'
        )
        
        # 分离好和不好的关卡
        self.good_levels = self.df[self.df['流量分类'] == '好'].copy()
        self.bad_levels = self.df[self.df['流量分类'] != '好'].copy()
        
        # 显示每个关卡的匹配情况
        print(f"\n关卡匹配详情:")
        for idx, row in self.df.iterrows():
            level_name = row['关卡名称']
            normalized = normalize_level_name(level_name)
            matched = normalized in good_levels_normalized
            print(f"  关卡: {level_name} -> 标准化: {normalized} -> 匹配: {'✓' if matched else '✗'}")
        
        print(f"\n分类结果:")
        print(f"流量好的关卡: {len(self.good_levels)} 个")
        if len(self.good_levels) > 0:
            print(f"  关卡列表: {self.good_levels['关卡名称'].tolist()}")
        print(f"流量不好的关卡: {len(self.bad_levels)} 个")
        if len(self.bad_levels) > 0:
            print(f"  关卡列表: {self.bad_levels['关卡名称'].tolist()}")
        
        return self.good_levels, self.bad_levels
    
    def analyze_features(self, output_file: Optional[str] = None):
        """
        分析特征差异
        
        Args:
            output_file: 输出Excel文件路径（可选）
        """
        if self.good_levels is None or self.bad_levels is None:
            raise ValueError("请先分类关卡")
        
        if self.good_levels.empty or self.bad_levels.empty:
            print("警告: 好关卡或不好关卡数量为0，无法进行对比分析")
            return
        
        print("\n" + "="*60)
        print("=== 流量好 vs 流量不好关卡特征对比分析 ===")
        print("="*60)
        
        # 获取数值特征列
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除非特征列
        exclude_cols = ['图片宽度', '图片高度', '总像素数']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"\n分析 {len(feature_cols)} 个数值特征...\n")
        
        # 计算统计信息
        results = []
        
        for col in feature_cols:
            good_values = self.good_levels[col].dropna()
            bad_values = self.bad_levels[col].dropna()
            
            if len(good_values) == 0 or len(bad_values) == 0:
                continue
            
            good_mean = good_values.mean()
            bad_mean = bad_values.mean()
            good_std = good_values.std()
            bad_std = bad_values.std()
            
            diff = good_mean - bad_mean
            diff_percent = (diff / bad_mean * 100) if bad_mean != 0 else 0
            
            # 计算显著性（简单的t检验）
            from scipy import stats
            try:
                t_stat, p_value = stats.ttest_ind(good_values, bad_values)
                is_significant = p_value < 0.05
            except:
                t_stat, p_value = None, None
                is_significant = False
            
            results.append({
                '特征名称': col,
                '好关卡均值': round(good_mean, 3),
                '好关卡标准差': round(good_std, 3),
                '不好关卡均值': round(bad_mean, 3),
                '不好关卡标准差': round(bad_std, 3),
                '差异': round(diff, 3),
                '差异百分比(%)': round(diff_percent, 2),
                't统计量': round(t_stat, 3) if t_stat is not None else None,
                'p值': round(p_value, 4) if p_value is not None else None,
                '是否显著(p<0.05)': '是' if is_significant else '否'
            })
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 按差异绝对值排序
        results_df = results_df.sort_values('差异', key=abs, ascending=False)
        
        # 显示结果
        print("特征差异排序（按差异绝对值）:")
        print("-"*60)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        
        # 显示前20个差异最大的特征
        print(results_df.head(20).to_string(index=False))
        
        # 显示显著差异的特征
        significant_features = results_df[results_df['是否显著(p<0.05)'] == '是']
        if not significant_features.empty:
            print(f"\n\n显著差异的特征 (p < 0.05): {len(significant_features)} 个")
            print("-"*60)
            print(significant_features[['特征名称', '好关卡均值', '不好关卡均值', '差异', '差异百分比(%)', 'p值']].to_string(index=False))
        
        # 保存结果
        if output_file:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 保存详细对比结果
                results_df.to_excel(writer, sheet_name='特征对比', index=False)
                
                # 保存好关卡数据
                self.good_levels.to_excel(writer, sheet_name='流量好关卡', index=False)
                
                # 保存不好关卡数据
                self.bad_levels.to_excel(writer, sheet_name='流量不好关卡', index=False)
                
                # 保存汇总统计
                summary = {
                    '统计项': ['流量好关卡数量', '流量不好关卡数量', '总关卡数', '分析特征数', '显著差异特征数'],
                    '数值': [
                        len(self.good_levels),
                        len(self.bad_levels),
                        len(self.df),
                        len(feature_cols),
                        len(significant_features)
                    ]
                }
                pd.DataFrame(summary).to_excel(writer, sheet_name='汇总统计', index=False)
            
            print(f"\n\n分析结果已保存到: {output_file}")
        
        return results_df
    
    def print_summary_statistics(self):
        """打印汇总统计信息"""
        if self.good_levels is None or self.bad_levels is None:
            return
        
        print("\n" + "="*60)
        print("=== 汇总统计 ===")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['图片宽度', '图片高度', '总像素数']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"\n流量好关卡 ({len(self.good_levels)} 个):")
        print(f"  平均特征值范围: {self.good_levels[feature_cols].mean().min():.2f} ~ {self.good_levels[feature_cols].mean().max():.2f}")
        
        print(f"\n流量不好关卡 ({len(self.bad_levels)} 个):")
        print(f"  平均特征值范围: {self.bad_levels[feature_cols].mean().min():.2f} ~ {self.bad_levels[feature_cols].mean().max():.2f}")


if __name__ == "__main__":
    # ========== 配置参数（在这里修改） ==========
    # 特征文件路径
    features_file = "level_features.xlsx"
    # 流量好的关卡列表（直接在这里修改，支持带或不带"level"前缀）
    good_levels_list = ["level3008", "level3006","level3005","level14","level5","level10","level26","level27","level30","level31","level32","level34","level36","level37","level38","level39","level40","level41","level42","level43","level45","level47"]  # 修改这里的关卡列表
    
    # 输出文件路径
    output_file = "level_analysis_result.xlsx"
    # ==========================================
    
    try:
        # 创建分析器
        analyzer = LevelFeatureAnalyzer(features_file)
        
        # 加载特征数据
        analyzer.load_features()
        
        # 分类关卡（会自动标准化关卡名称）
        analyzer.classify_levels(good_levels_list)
        
        # 分析特征
        analyzer.analyze_features(output_file)
        
        # 打印汇总统计
        analyzer.print_summary_statistics()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保:")
        print("1. level_features.xlsx 文件存在")
        print("2. 在代码中正确设置了流量好关卡列表")

