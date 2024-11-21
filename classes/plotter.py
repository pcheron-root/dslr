class Plotter:
    def __init__(self, pd_data):
        self.pd_data = pd_data
        self.colormap = {
            "Ravenclaw": "tab:blue",
            "Slytherin": "tab:green",
            "Gryffindor": "tab:red",
            "Hufflepuff": "gold",
        }
