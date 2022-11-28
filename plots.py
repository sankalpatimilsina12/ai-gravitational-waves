from prepare_data import Data
import plotly.graph_objects as go


class Plot:
    def __init__(self):
        self.fig = go.Figure()

    def generate_frequency_plot(self):
        print('Reading training files....')
        data_model = Data()
        frequencies, labels = data_model.generate_frequency_data()

        print('Preparing data....')
        positives = []
        negatives = []
        for i, label in enumerate(labels):
            if label == 1:
                positives.append(frequencies[i])
            else:
                negatives.append(frequencies[i])

        print('Plotting frequency data....')
        samples = [i + 1 for i in range(len(frequencies))]
        self.fig.add_trace(go.Scatter(
            x=positives,
            y=samples,
            marker=dict(color='royalblue', size=7),
            mode='markers',
            name='Positives'
        ))
        self.fig.add_trace(go.Scatter(
            x=negatives,
            y=samples,
            marker=dict(color='crimson', size=7),
            mode='markers',
            name='Negatives'
        ))

        self.fig.update_layout(title='Frequency vs Training Sample',
                               xaxis_title='Frequency (in Hertz)', yaxis_title='Sample Number')
        self.fig.show()


if __name__ == '__main__':
    plot = Plot()
    plot.generate_frequency_plot()
