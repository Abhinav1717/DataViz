import matplotlib.pyplot as plt
from matplotlib import style
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
# Line Graph plot
def get_line_graph(data, x_column, y_column,style):
    plt.style.use(style)
    if (x_column == " "):
        plt.plot(data[y_column[0]])
    else:
        x_column_data = data[x_column]
        y_column_data = data[y_column]
        zippedXY = list(zip(x_column_data,y_column_data))
        zippedXY.sort(key=lambda x: x[0])
        x_column_data = [x for (x,y) in zippedXY]
        y_column_data = [y for (x,y) in zippedXY]
        plt.plot(x_column_data,y_column_data, label=y_column,mew=2, linewidth=2)
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.xticks(rotation=90)
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot

# Bar graph
def get_bar_graph(data, x_column, y_column,style):
    plt.style.use(style)
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
# Scatter Plot
def get_scatter_plot(data, x_column, y_column,style):
    plt.style.use(style)
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
# Hexbin Plot
def get_hexbin_plot(data,x_column,y_column,style):
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.hexbin(data[x_column], data[y_column],gridsize=20)
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot

#Stem Plot functionality

def get_stem_plot(data, x_column, y_column,style):
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.stem(data[x_column], data[y_column])
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot

# Histogram 2D plot

def get_hist2d_plot(data, x_column, y_column,style):
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.hist2d(data[x_column], data[y_column])
        plt.xlabel(x_column)
    # plt.grid(b='on')
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot

# Triplot 
def get_triplot_plot(data, x_column, y_column,style):
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.triplot(data[x_column], data[y_column])
        plt.xlabel(x_column)
    # plt.grid(b='on')
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot