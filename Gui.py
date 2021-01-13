
from tkinter import *
import Config
import Chart_plotter


class Gui:
    def __init__(self):
        self.root = Tk()
        self.root.title("Currency prediction")
        self.info = Label(self.root, text="Exchange rates from 2020", padx=10, pady=10, font='Helvetica 14 bold')
        self.currency_label = Label(self.root, text="Currency", padx=30, pady=10, font='Helvetica 14 bold')
        self.currency = StringVar()
        self.currency.set("USD")
        self.drop_menu = OptionMenu(self.root, self.currency, 'USD', 'EUR', 'GBP')
        self.label_training_percent = Label(self.root, text="Percentage of data for training", padx=30, pady=10, font='Helvetica 14 bold')
        self.input_training_percent = Entry(self.root, width=6, borderwidth=5)
        self.input_training_percent.insert(0, "70")
        self.label_arima = Label(self.root, text="ARIMA Config(p,d,q)", padx=30, pady=10, font='Helvetica 14 bold')
        self.input_p = Entry(self.root, width=6, borderwidth=5)
        self.input_p.insert(0, "5")
        self.input_d = Entry(self.root, width=6, borderwidth=5)
        self.input_d.insert(0, "1")
        self.input_q = Entry(self.root, width=6, borderwidth=5)
        self.input_q.insert(0, "0")
        self.label_rnn = Label(self.root, text="RNN Config(Previous points, epochs, batch_size)", padx=30, pady=10, font='Helvetica 14 bold')
        self.input_previous_data_points = Entry(self.root, width=6, borderwidth=5)
        self.input_previous_data_points.insert(0, "3")
        self.input_epochs = Entry(self.root, width=6, borderwidth=5)
        self.input_epochs.insert(0, "50")
        self.input_batch_size = Entry(self.root, width=6, borderwidth=5)
        self.input_batch_size.insert(0, "2")
        self.start = Button(self.root, text="Start", padx=350, pady=20, command=self.start_counting, font='Helvetica 18 bold')


        self.info.grid(row=0, column=0, columnspan=2)
        self.currency_label.grid(row=1, column=0)
        self.drop_menu.grid(row=1, column=1)
        self.label_training_percent.grid(row=2, column=0)
        self.input_training_percent.grid(row=2, column=1)
        self.label_arima.grid(row=3, column=0)
        self.input_p.grid(row=4, column=0)
        self.input_d.grid(row=5, column=0)
        self.input_q.grid(row=6, column=0)
        self.label_rnn.grid(row=3, column=1)
        self.input_previous_data_points.grid(row=4, column=1)
        self.input_epochs.grid(row=5, column=1)
        self.input_batch_size.grid(row=6, column=1)
        self.start.grid(row=7, columnspan=2)
        self.root.mainloop()

    def start_counting(self):
        Config.currency = self.currency.get()
        Config.training_percent = int(self.input_training_percent.get())/100
        Config.p = int(self.input_p.get())
        Config.d = int(self.input_d.get())
        Config.q = int(self.input_q.get())
        Config.previous_data_points = int(self.input_previous_data_points.get())
        Config.batch_size = int(self.input_batch_size.get())
        Config.epochs = int(self.input_epochs.get())
        Chart_plotter.start()

        self.root.destroy()
