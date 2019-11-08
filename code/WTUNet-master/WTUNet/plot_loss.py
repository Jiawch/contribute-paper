import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('loss.pdf')
# 读取tensorboard保存的acc、loss数据
df = pd.read_excel('loss.xlsx')
middle_val_loss =df['middle-loss']
base_val_loss =df['base-loss']
# 本次实验一共训练250个epoch
epochs = 129
# 绘制训练的acc曲线图
plt.plot(range(epochs), middle_val_loss, '-.', label='TF-Attention-Net-Middle')
plt.plot(range(epochs), base_val_loss, label='TF-Attention-Net-Base')
# 绘制标题
plt.title('Validation-Loss')
# 绘制标签
plt.legend()
# 输出图像
pdf.savefig()
plt.close()
pdf.close()