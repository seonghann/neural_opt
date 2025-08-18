import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse


def setup_plot_style(base_font_size=14):
    """전체 플롯 스타일 설정을 한 곳에서 관리합니다.

    Args:
        base_font_size (int): 기본 폰트 크기. 다른 요소들은 이에 비례하여 조정됩니다.
    """
    # 비례 계산을 위한 스케일 팩터
    scale_factors = {
        "title": 1.3,  # 제목은 기본 크기의 130%
        "label": 1.15,  # 축 레이블은 115%
        "tick": 1.0,  # 틱 레이블은 100%
        "legend": 0.85,  # 범례는 85%
        "text": 1.2,  # 텍스트 박스 (correlation 표시)는 120%
    }

    plt.rcParams.update(
        {
            # 기본 폰트 크기
            "font.size": base_font_size,
            # 제목 관련
            "axes.titlesize": base_font_size * scale_factors["title"],
            "figure.titlesize": base_font_size * scale_factors["title"],
            # 축 레이블
            "axes.labelsize": base_font_size * scale_factors["label"],
            # 틱 레이블
            "xtick.labelsize": base_font_size * scale_factors["tick"],
            "ytick.labelsize": base_font_size * scale_factors["tick"],
            # 범례
            "legend.fontsize": base_font_size * scale_factors["legend"],
            "legend.title_fontsize": base_font_size * scale_factors["legend"],
            # 추가 설정 (선택사항)
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.autolayout": True,
        }
    )

    return scale_factors


def plot_scatter_with_correlation(
    df, x_col, y_col, figsize=(6, 6), save_path=None, log_scale=False, font_size=14, visualize=True
):
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
        그래프 크기 (default: (6, 6))
    save_path : str
        그래프를 저장할 경로 (None이면 저장하지 않음)
    log_scale : bool
        로그 스케일 사용 여부
    font_size : int
        기본 폰트 크기 (default: 14)

    Returns:
    --------
    correlation : float
        Pearson correlation coefficient
    p_value : float
        p-value for the correlation
    """

    # 플롯 스타일 설정
    scale_factors = setup_plot_style(font_size)

    # 데이터 준비
    x = df[x_col].values
    y = df[y_col].values

    # NaN 값 제거
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    # 로그 스케일인 경우 0 이하 값 제거
    if log_scale:
        positive_mask = (x_clean > 0) & (y_clean > 0)
        x_clean = x_clean[positive_mask]
        y_clean = y_clean[positive_mask]

    # Pearson correlation 계산
    correlation, p_value = stats.pearsonr(x_clean, y_clean)

    # 그래프 그리기
    plt.figure(figsize=figsize)

    # Scatter plot
    # plt.scatter(x_clean, y_clean, alpha=0.6, s=10, color='blue', marker='.')
    plt.scatter(x_clean, y_clean, marker=".")

    # 로그 스케일 설정
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    # 그래프 스타일링
    if "rmsd" in x_col.lower():
        xlabel = "RMSD ($\AA$)"
    elif "dmae" in x_col.lower():
        xlabel = "D-MAE ($\AA$)"
    elif "q_norm" in x_col.lower():
        # xlabel = "$\|q_0 - q_t\|_2$"
        xlabel = "$\|\mathbf{q}_0 - \mathbf{q}_t\|_2$"
    plt.xlabel(xlabel)
    plt.ylabel("$|\Delta E|$ (kcal/mol)")

    # correlation을 그래프 왼쪽 위에 표시
    text_size = font_size * scale_factors["text"]
    plt.text(
        0.05,
        0.95,
        f"r={correlation:.2f}",
        transform=plt.gca().transAxes,
        fontsize=text_size,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )

    # 그리드 추가
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()

    # 그래프 저장 (옵션)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if visualize:
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
    parser = argparse.ArgumentParser(
        description="Create scatter plot and calculate Pearson correlation"
    )
    parser.add_argument(
        "-x",
        "--x-axis",
        type=str,
        required=True,
        help="Column name for x-axis (e.g., rmsd_xT)",
    )
    parser.add_argument(
        "-y",
        "--y-axis",
        type=str,
        required=True,
        help="Column name for y-axis (e.g., delta_E)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="denoise.random.with_energy.csv",
        help="CSV file path (default: denoise.random.with_energy.csv)",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="Save plot to file (e.g., plot.png, plot.svg)",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[6, 5.5],
        help="Figure size in inches (default: 6 5.5)",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="List all available columns in the dataframe",
    )
    parser.add_argument(
        "--log",
        "--log-scale",
        action="store_true",
        default=False,
        help="Use log scale for both axes",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=20,
        help="Base font size for all plot elements (default: 20)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the plot interactively",
    )

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
            print("Available columns:", ", ".join(df.columns))
            return

        if args.y_axis not in df.columns:
            print(f"Error: Column '{args.y_axis}' not found in dataframe")
            print("Available columns:", ", ".join(df.columns))
            return

        # delta_E 컬럼인 경우 절댓값 처리
        if "delta_E" in args.y_axis or "energy" in args.y_axis.lower():
            df[args.y_axis] = df[args.y_axis].abs()

        # Scatter plot 생성 및 상관계수 계산
        figsize = tuple(args.figsize)
        corr, p_val = plot_scatter_with_correlation(
            df,
            args.x_axis,
            args.y_axis,
            figsize=figsize,
            save_path=args.save,
            log_scale=args.log,
            font_size=args.font_size,
            visualize=args.visualize,
        )

    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
