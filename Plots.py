import urllib.request
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Plots:

    def draw_plots(self, url):
        df = pd.read_json(url)
        pd.set_option('display.max_columns', None)
        print(df.head())
        print(df.info())
        print(df.describe(exclude=['O']))
        self.plot_heatmap(df)
        self.plot_corners(df)
        self.plot_corners_hist(df)
        self.plot_scatterplot(df)
        self.plot_boxplot(df)

    def plot_heatmap(self, df):
        sns.set(font_scale=1)
        plt.figure(figsize=(12, 8))
        plt.title("Heatmap")
        sns.heatmap(
            df.corr(numeric_only=True),
            cmap='RdBu_r',  # задаёт цветовую схему
            annot=True,  # рисует значения внутри ячеек
            vmin=-1, vmax=1)  # указывает начало цветовых кодов от -1 до 1.
        plt.savefig("plots/heatmap.png")
        plt.show()

    def plot_corners(self, df):
        plt.scatter(df["gt_corners"], df["rb_corners"])
        plt.xlabel("Ground truth number of corners")
        plt.ylabel("Number of corners found by the model")
        plt.savefig("plots/corners.png")
        plt.show()


    def plot_corners_hist(self, df):
        plt.hist(df["gt_corners"])
        plt.title("Ground truth number of corners")
        plt.xlabel("Number of corners")
        plt.ylabel("Number of rooms")
        plt.savefig("plots/gt_corners_hist.png")
        plt.show()
        plt.hist(df["rb_corners"])
        plt.title("Number of corners found by the model")
        plt.xlabel("Number of corners")
        plt.ylabel("Number of rooms")
        plt.savefig("plots/rb_corners_hist.png")
        plt.show()

    def plot_scatterplot(self, df):
        sns.scatterplot(data=df, x="max", y="min", hue="gt_corners", palette="Set1")
        plt.title("Scatterplot gt_corners", fontsize=18)
        plt.savefig("plots/scatterplot.png")
        plt.show()
        face_grid = sns.FacetGrid(df, col="gt_corners")
        face_grid.map(sns.scatterplot, "max", "min")
        plt.savefig("plots/facetgrid_scatterplot.png")
        plt.show()

    def plot_boxplot(self, df):
        sns.boxplot(
            x='gt_corners',
            y='mean',
            data=df,
            palette="Set1")
        plt.title("Boxplot")
        plt.savefig("plots/boxplot.png")
        plt.show()
        sns.barplot(data=df, x='gt_corners', y='mean', palette="Set1")
        plt.title("Barplot")
        plt.savefig("plots/barplot.png")
        plt.show()

