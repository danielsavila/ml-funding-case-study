import matplotlib.pyplot as plt

#visuals are helpful for later diagnosis of model outputs
def visualizations(true, predicted, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(true, predicted, alpha=0.6, color='blue', edgecolor='k')
        plt.plot([true.min(), true.max()],
                [true.min(), true.max()],
                'r--', linewidth=2)  # reference line y=x
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{title}')
        plt.grid(True)
        
        return plt.show()
