import matplotlib.pyplot as plt
from matplotlib import style
from base64 import b64decode
from base64 import b64encode
from io import BytesIO

def plot_buffer_to_utf8(plot_buffer):
    """
    Encodes image file into a base64 encoding

    Parameters:
    plot_buffer (BytesIO Object: Plot image is saved in this file

    Returns:
    Base64 encoding of the input image.

    """
    plot_buffer.seek(0)
    plot_png = plot_buffer.getvalue()
    plot_base64 = b64encode(plot_png)
    plot_utf8 = plot_base64.decode('utf-8')

    return plot_utf8
    
# Line Graph plot
def get_line_graph(data, x_column, y_column,style):
    """
    Generates Line Graph plot

    Parameters:
    data : 2D list of data file
    x_column : The parameter to be used for X axis values of the plot
    y_column : The parameter to be used for Y axis plot points
    style : Which Matplotlib style to use
    Returns:
    Base64 encoding of the generated plot

    """
    plt.clf()
    plt.close()
    plt.style.use(style)
    if (x_column == " "):
        plt.plot(data[y_column[0]])
    else:
        x_column_data = data[x_column]
        y_column_data = data[y_column]
        zippedXY = list(zip(x_column_data,y_column_data))
        # Sort the data based on X-axis values to generate a cleaner image for the line graph
        zippedXY.sort(key=lambda x: x[0])
        x_column_data = [x for (x,y) in zippedXY]
        y_column_data = [y for (x,y) in zippedXY]
        plt.plot(x_column_data,y_column_data, label=y_column,mew=2, linewidth=2)
        plt.xlabel(x_column)
    plt.grid(b='on')
    # Tilt the X axis tick values for better visibility
    plt.xticks(rotation=90)
    plt.ylabel(y_column)
    # plt.ticklabel_format(style="plain")
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png",bbox_inches="tight")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    plot_buffer.close()
    return encoded_plot

# Bar graph
def get_bar_graph(data, x_column, y_column,style):
    """
    Generates Bar Graph plot

    Parameters:
    data : 2D list of data file
    x_column : The parameter to be used for X axis values of the plot
    y_column : The parameter to be used for Y axis plot points
    style : Which Matplotlib style to use
    Returns:
    Base64 encoding of the generated plot

    """
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.bar(data[x_column], data[y_column])
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.xticks(rotation=90)
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png",bbox_inches="tight")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot
# Scatter Plot
def get_scatter_plot(data, x_column, y_column,style):
    """
    Generates Scatter plot graph

    Parameters:
    data : 2D list of data file
    x_column : The parameter to be used for X axis values of the plot
    y_column : The parameter to be used for Y axis plot points
    style : Which Matplotlib style to use
    Returns:
    Base64 encoding of the generated plot

    """
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.scatter(data[x_column], data[y_column])
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.xticks(rotation=90)
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png",bbox_inches="tight")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot
# Hexbin Plot
def get_hexbin_plot(data,x_column,y_column,style):
    """
    Generates Hexbin plot

    Parameters:
    data : 2D list of data file
    x_column : The parameter to be used for X axis values of the plot
    y_column : The parameter to be used for Y axis plot points
    style : Which Matplotlib style to use
    Returns:
    Base64 encoding of the generated plot

    """
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.hexbin(data[x_column], data[y_column],gridsize=20)
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.xticks(rotation=90)
    plt.ylabel(y_column)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png",bbox_inches="tight")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot

#Stem Plot functionality

def get_stem_plot(data, x_column, y_column,style):
    """
    Generates Stem plot

    Parameters:
    data : 2D list of data file
    x_column : The parameter to be used for X axis values of the plot
    y_column : The parameter to be used for Y axis plot points
    style : Which Matplotlib style to use
    Returns:
    Base64 encoding of the generated plot

    """
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.stem(data[x_column], data[y_column])
        plt.xlabel(x_column)
    plt.grid(b='on')
    plt.ylabel(y_column)
    plt.xticks(rotation=90)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png",bbox_inches="tight")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot

# Histogram 2D plot

def get_hist2d_plot(data, x_column, y_column,style):
    """
    Generates 2D Histogram for the given data

    Parameters:
    data : 2D list of data file
    x_column : The parameter to be used for X axis values of the plot
    y_column : The parameter to be used for Y axis plot points
    style : Which Matplotlib style to use
    Returns:
    Base64 encoding of the generated plot

    """
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.hist2d(data[x_column], data[y_column])
        plt.xlabel(x_column)
    # plt.grid(b='on')
    plt.ylabel(y_column)
    plt.xticks(rotation=90)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png",bbox_inches="tight")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot

# Triplot 
def get_triplot_plot(data, x_column, y_column,style):
    """
    Generates Triplot Graph for the given data

    Parameters:
    data : 2D list of data file
    x_column : The parameter to be used for X axis values of the plot
    y_column : The parameter to be used for Y axis plot points
    style : Which Matplotlib style to use
    Returns:
    Base64 encoding of the generated plot

    """
    plt.style.use(style)
    if (x_column == " "):
        return None
    else:
        plt.triplot(data[x_column], data[y_column])
        plt.xlabel(x_column)
    # plt.grid(b='on')
    plt.ylabel(y_column)
    plt.xticks(rotation=90)
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png",bbox_inches="tight")
    plt.close()

    encoded_plot = plot_buffer_to_utf8(plot_buffer)
    return encoded_plot