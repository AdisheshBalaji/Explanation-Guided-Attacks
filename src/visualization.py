import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='muted', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

def plot_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('COMPAS Dataset — Overview', fontsize=14, fontweight='bold')

    # Target distribution
    counts = df['two_year_recid'].value_counts()
    axes[0].bar(['No Recidivism', 'Recidivism'], counts.values, color=['#4C9BE8', '#E85C4C'])
    axes[0].set_title('Target: Two-Year Recidivism')

    # Race distribution
    race_counts = df['race'].value_counts()
    axes[1].bar(race_counts.index, race_counts.values, color=['#A78BFA', '#34D399'])
    axes[1].set_title('Race Distribution (Filtered)')

    # Sex distribution
    sex_counts = df['sex'].value_counts()
    axes[2].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', colors=['#60A5FA', '#F9A8D4'])
    axes[2].set_title('Sex Distribution')

    plt.tight_layout()
    plt.savefig('viz_01_overview.png', bbox_inches='tight')
    plt.show()