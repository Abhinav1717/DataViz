import matplotlib.pyplot as plt
from base64 import b64decode
from base64 import b64encode
from io import BytesIO

def plot_buffer_to_utf8(plot_buffer):
    plot_buffer.seek(0)
    plot_png = plot_buffer.getvalue()
    plot_buffer.close()

    plot_base64 = b64encode(plot_png)
    plot_utf8 = plot_base64.decode('utf-8')

    return plot_utf8


def get_line_graph(data, x_column, y_column):
    if (x_column == " "):
        plt.plot(data[y_column])
    else:
        plt.plot(data[x_column], data[y_column])
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot


def get_bar_graph(data, x_column, y_column):
    if (x_column == " "):
        return None
    else:
        plt.bar(data[x_column], data[y_column])
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot

def get_scatter_plot(data, x_column, y_column):
    if (x_column == " "):
        return None
    else:
        plt.scatter(data[x_column], data[y_column])
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot

def get_hexbin_plot(data,x_column,y_column):
    if (x_column == " "):
        return None
    else:
        plt.hexbin(data[x_column], data[y_column])
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot
