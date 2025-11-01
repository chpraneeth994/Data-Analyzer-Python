import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.summary_stats = {}
        
    def load_sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sales = np.random.normal(1000, 200, 100) + np.sin(np.arange(100) * 0.1) * 100
        customers = np.random.poisson(50, 100)
        products = np.random.choice(['Electronics', 'Clothing', 'Books', 'Food'], 100)        
        self.data = pd.DataFrame({
            'Date': dates,
            'Sales': sales,
            'Customers': customers,
            'Product_Category': products
        })
        
        print("✅ Sample data loaded successfully!")
        return self.data
    
    def basic_statistics(self):
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.summary_stats[col] = {
                'Mean': np.mean(self.data[col]),
                'Median': np.median(self.data[col]),
                'Std': np.std(self.data[col]),
                'Min': np.min(self.data[col]),
                'Max': np.max(self.data[col]),
                'Q1': np.percentile(self.data[col], 25),
                'Q3': np.percentile(self.data[col], 75)
            }
        
        print("\n📊 Basic Statistics:")
        print("=" * 50)
        for col, stats in self.summary_stats.items():
            print(f"\n{col}:")
            for stat, value in stats.items():
                print(f"  {stat}: {value:.2f}")
    
    def correlation_analysis(self):
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = np.corrcoef(numeric_data.T)
        
        print("\n🔗 Correlation Analysis:")
        print("=" * 50)
        print("Correlation Matrix:")
        print(correlation_matrix)
        for i in range(len(numeric_data.columns)):
            for j in range(i+1, len(numeric_data.columns)):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.5:
                    print(f"\nStrong correlation ({corr:.3f}) between {numeric_data.columns[i]} and {numeric_data.columns[j]}")
    
    def trend_analysis(self):
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        if 'Date' not in self.data.columns:
            print("❌ No date column found for trend analysis.")
            return
        window_size = 7
        sales_ma = np.convolve(self.data['Sales'], np.ones(window_size)/window_size, mode='valid')
        
        print(f"\n📈 Trend Analysis (Moving Average - {window_size} days):")
        print("=" * 50)
        print(f"Overall trend: {'Increasing' if sales_ma[-1] > sales_ma[0] else 'Decreasing'}")
        print(f"Average daily sales: {np.mean(self.data['Sales']):.2f}")
        print(f"Sales volatility: {np.std(self.data['Sales']):.2f}")
    
    def category_analysis(self):
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        if 'Product_Category' not in self.data.columns:
            print("❌ No category column found for analysis.")
            return
        
        print("\n📋 Category Analysis:")
        print("=" * 50)
        
        category_stats = self.data.groupby('Product_Category').agg({
            'Sales': ['mean', 'sum', 'count'],
            'Customers': ['mean', 'sum']
        }).round(2)
        
        print(category_stats)
        
        best_category = category_stats[('Sales', 'sum')].idxmax()
        print(f"\n🏆 Best performing category: {best_category}")
    
    def generate_visualizations(self):
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Analysis Dashboard - Visualizations', fontsize=16)
        
        axes[0, 0].plot(self.data['Date'], self.data['Sales'], 'b-', linewidth=2)
        axes[0, 0].set_title('Sales Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sales')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].hist(self.data['Sales'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Sales Distribution')
        axes[0, 1].set_xlabel('Sales')
        axes[0, 1].set_ylabel('Frequency')
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = np.corrcoef(numeric_data.T)
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('Correlation Heatmap')
        axes[1, 0].set_xticks(range(len(numeric_data.columns)))
        axes[1, 0].set_yticks(range(len(numeric_data.columns)))
        axes[1, 0].set_xticklabels(numeric_data.columns, rotation=45)
        axes[1, 0].set_yticklabels(numeric_data.columns)
        plt.colorbar(im, ax=axes[1, 0])
        
        if 'Product_Category' in self.data.columns:
            category_sales = self.data.groupby('Product_Category')['Sales'].sum()
            axes[1, 1].bar(category_sales.index, category_sales.values, color=['red', 'blue', 'green', 'orange'])
            axes[1, 1].set_title('Sales by Category')
            axes[1, 1].set_xlabel('Product Category')
            axes[1, 1].set_ylabel('Total Sales')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        plt.savefig('outputs/data_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualizations saved to 'outputs/data_analysis_dashboard.png'")
        plt.show()
    
    def export_report(self):
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        
        with open('outputs/analysis_report.txt', 'w') as f:
            f.write("DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total records: {len(self.data)}\n")
            f.write(f"Columns: {', '.join(self.data.columns)}\n\n")
            
            f.write("BASIC STATISTICS\n")
            f.write("-" * 20 + "\n")
            for col, stats in self.summary_stats.items():
                f.write(f"\n{col}:\n")
                for stat, value in stats.items():
                    f.write(f"  {stat}: {value:.2f}\n")
        
        print("📄 Analysis report exported to 'outputs/analysis_report.txt'")

def main():

    print("🚀 Data Analysis Dashboard")
    print("=" * 50) 
    analyzer = DataAnalyzer()
    analyzer.load_sample_data()
    analyzer.basic_statistics()
    analyzer.correlation_analysis()
    analyzer.trend_analysis()
    analyzer.category_analysis()
    analyzer.generate_visualizations()
    analyzer.export_report()
    
    print("\n✅ Data analysis completed successfully!")
    print("📁 Check the 'outputs' folder for generated files.")

if __name__ == "__main__":
    main() 