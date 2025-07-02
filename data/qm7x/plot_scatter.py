import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse

def plot_scatter_with_correlation(df, x_col, y_col, figsize=(10, 8), save_path=None):
    """
    Scatter plot을 그리고 Pearson correlation을 계산하는 함수

    Parameters:
    -----------
    df : pandas.DataFrame
        데이터프레임
    x_col : str
        x축에 사용할 컬럼명
    y_col : str
        y축에 사용할 컬럼명
    figsize : tuple
        그래프 크기 (default: (10, 8))
    save_path : str
        그래프를 저장할 경로 (None이면 저장하지 않음)

    Returns:
    --------
    correlation : float
        Pearson correlation coefficient
    p_value : float
        p-value for the correlation
    """

    # 데이터 준비
    x = df[x_col]
    y = df[y_col]

    # NaN 값 제거
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    # Pearson correlation 계산
    correlation, p_value = stats.pearsonr(x_clean, y_clean)

    # 그래프 그리기
    plt.figure(figsize=figsize)

    # Scatter plot
    plt.scatter(x_clean, y_clean, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # 회귀선 추가
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(x_clean), p(np.sort(x_clean)), "r--", alpha=0.8, linewidth=2)

    # 그래프 스타일링
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Correlation 정보를 그래프에 표시
    textstr = f'Pearson r = {correlation:.4f}\np-value = {p_value:.4e}\nn = {len(x_clean)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    
    # 그래프 저장 (옵션)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

    # 결과 출력
    print(f"\n=== Correlation Analysis Results ===")
    print(f"X-axis: {x_col}")
    print(f"Y-axis: {y_col}")
    print(f"Pearson Correlation Coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"Number of data points: {len(x_clean)}")
    
    # 상관관계 강도 해석
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if correlation > 0 else "negative"
    print(f"Interpretation: {strength} {direction} correlation")

    return correlation, p_value


def main():
    # Argument parser 설정
    parser = argparse.ArgumentParser(description='Create scatter plot and calculate Pearson correlation')
    parser.add_argument('-x', '--x-axis', type=str, required=True,
                        help='Column name for x-axis (e.g., rmsd_xT)')
    parser.add_argument('-y', '--y-axis', type=str, required=True,
                        help='Column name for y-axis (e.g., delta_E)')
    parser.add_argument('-f', '--file', type=str, default='denoise.random.with_energy.csv',
                        help='CSV file path (default: denoise.random.with_energy.csv)')
    parser.add_argument('-s', '--save', type=str, default=None,
                        help='Save plot to file (e.g., plot.png)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 8],
                        help='Figure size in inches (default: 10 8)')
    parser.add_argument('--list-columns', action='store_true',
                        help='List all available columns in the dataframe')
    
    args = parser.parse_args()
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(args.file)
        print(f"Successfully loaded data from: {args.file}")
        print(f"Data shape: {df.shape}")
        
        # 컬럼 목록 출력 옵션
        if args.list_columns:
            print("\nAvailable columns:")
            for i, col in enumerate(df.columns):
                print(f"  {i+1:2d}. {col}")
            return
        
        # 지정된 컬럼이 존재하는지 확인
        if args.x_axis not in df.columns:
            print(f"Error: Column '{args.x_axis}' not found in dataframe")
            print("Available columns:", ', '.join(df.columns))
            return
        
        if args.y_axis not in df.columns:
            print(f"Error: Column '{args.y_axis}' not found in dataframe")
            print("Available columns:", ', '.join(df.columns))
            return
        
        # Scatter plot 생성 및 상관계수 계산
        figsize = tuple(args.figsize)
        corr, p_val = plot_scatter_with_correlation(
            df, 
            args.x_axis, 
            args.y_axis, 
            figsize=figsize,
            save_path=args.save
        )
        
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
