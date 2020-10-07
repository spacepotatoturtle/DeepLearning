import csv


class Image:

    """
    I used a class to convert the csv files into usable data to feed into the network inputs. It
    also has a display method for me to see the image. The display is not very elegant, but it was
    functional.
    """

    # Takes longer with higher row numbers.

    def __init__(self, file, num):
        with open(file, "rt", newline='') as f:
            r = csv.reader(f)
            for line_num, line in enumerate(r):
                if line_num >= num:
                    self.data = line
                    break

        self.inputs = []
        for i in range(1, len(self.data)):
            self.inputs.append(int(self.data[i]))

        self.outputs = []
        for i in range(10):
            if i == int(self.data[0]):
                self.outputs.append(1)
            else:
                self.outputs.append(0)

    def display(self):
        for j in range(28):
            print(''.join(f'{self.inputs[i+(j*28)]:3}' for i in range(28)))
