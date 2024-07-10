import tkinter as tk
from tkinter import messagebox, filedialog
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



data = {
    'Math': [],
    'Science': [],
    'English': [],
    'Pass/Fail': [],
}

# for key, values in data.items():
#     print(f"{key}: {values}")

def u(x):
    if abs(x) < 1e-5:
        x = 0.0
    return 1 if x >= 0 else 0
class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry('500x500')
        self.root.title('Perceptron')
        self.root.configure(background='#2e2e2e')
        self.learning_rate = None
        self.max_epochs = None
        self.weights = [0.3, -0.1, 0.4]
        self.theta = -200
        self.mse = []
        self.goal = None
        #self.weights = [random.uniform(-1, 1) for _ in range(3)]  # Initialize weights randomly
        #self.theta = random.uniform(-1, 1)  # Initialize theta randomly
        self.create_main_menu()

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                # Skip the header line
                next(file)

                for line in file:
                    math, science, english, pass_fail = line.strip().split(',')
                    data['Math'].append(float(math))
                    data['Science'].append(float(science))
                    data['English'].append(float(english))
                    data['Pass/Fail'].append(pass_fail.lower())

            # Split data into training and testing sets (80% training, 20% testing)
            combined_data = list(zip(data['Math'], data['Science'], data['English'], data['Pass/Fail']))
            random.shuffle(combined_data)
            train_size = int(0.8 * len(combined_data))

            train_data = combined_data[:train_size]
            test_data = combined_data[train_size:]

            # Assign 80% to training data
            data['Math'], data['Science'], data['English'], data['Pass/Fail'] = zip(*train_data)

            # Create separate dictionaries for testing data
            self.test_data = {
                'Math': [x[0] for x in test_data],
                'Science': [x[1] for x in test_data],
                'English': [x[2] for x in test_data],
                'Pass/Fail': [x[3] for x in test_data],
            }

            self.open_file_button.config(state=tk.DISABLED)

    def create_main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        main_menu_frame = tk.Frame(self.root, bg='#2e2e2e')
        main_menu_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.train_button = tk.Button(main_menu_frame, text="Train Perceptron", command=self.train_perceptron_menu,
                                      font=('Helvetica', 16), bg='#2e2e2e', fg='white')
        self.test_button = tk.Button(main_menu_frame, text="Test Perceptron", command=self.test_perceptron_menu,
                                     font=('Helvetica', 16), bg='#2e2e2e', fg='white')

        self.train_button.pack(pady=20)
        self.test_button.pack(pady=20)

    def train_perceptron_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.lr_label = tk.Label(self.root, text="Learning Rate:", font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.lr_entry = tk.Entry(self.root, font=('Helvetica', 14), bg='#cbcbcb', fg='black')

        self.epochs_label = tk.Label(self.root, text="Max Epochs:", font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.epochs_entry = tk.Entry(self.root, font=('Helvetica', 14), bg='#cbcbcb', fg='black')


        self.goal_label = tk.Label(self.root, text="Goal:", font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.goal_entry = tk.Entry(self.root, font=('Helvetica', 14), bg='#cbcbcb', fg='black')

        # self.science_label = tk.Label(self.root, text="Science:", font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        # self.science_entry = tk.Entry(self.root, font=('Helvetica', 14), bg='#cbcbcb', fg='black')
        #
        # self.english_label = tk.Label(self.root, text="English:", font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        # self.english_entry = tk.Entry(self.root, font=('Helvetica', 14), bg='#cbcbcb', fg='black')
        #
        # self.pass_fail_label = tk.Label(self.root, text="Pass/Fail:", font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        # self.pass_fail_entry = tk.Entry(self.root, font=('Helvetica', 14), bg='#cbcbcb', fg='black')

        self.lr_label.pack(pady=5)
        self.lr_entry.pack(pady=5)

        self.epochs_label.pack(pady=5)
        self.epochs_entry.pack(pady=5)

        self.goal_label.pack(pady=5)
        self.goal_entry.pack(pady=5)

        # Button to open file dialog
        self.open_file_button = tk.Button(self.root, text="Open file.txt", command=self.open_file,
                                          font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.open_file_button.pack(pady=20)

        if self.learning_rate is not None and self.max_epochs is not None and self.goal is not None:
            self.lr_entry.insert(0, str(self.learning_rate))
            self.epochs_entry.insert(0, str(self.max_epochs))
            self.goal_entry.insert(0, str(self.goal))
            # print(self.learning_rate, self.max_epochs, self.goal)
            self.lr_entry.config(state='disabled')
            self.epochs_entry.config(state='disabled')
            self.open_file_button.config(state=tk.DISABLED)
            self.goal_entry.config(state='disabled')

        # self.science_label.pack(pady=5)
        # self.science_entry.pack(pady=5)
        #
        # self.english_label.pack(pady=5)
        # self.english_entry.pack(pady=5)
        #
        # self.pass_fail_label.pack(pady=5)
        # self.pass_fail_entry.pack(pady=5)

        self.lr_entry.bind("<Return>", lambda event: self.epochs_entry.focus_set())
        self.epochs_entry.bind("<Return>", lambda event: self.goal_entry.focus_set())

        # self.math_entry.bind("<Return>", lambda event: self.science_entry.focus_set())
        # self.science_entry.bind("<Return>", lambda event: self.english_entry.focus_set())
        # self.english_entry.bind("<Return>", lambda event: self.pass_fail_entry.focus_set())
        # self.pass_fail_entry.bind("<Return>", lambda event: self.save_data())


        self.save_button = tk.Button(self.root, text="Save", command=self.train_perceptron, font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.back_button = tk.Button(self.root, text="Back", command=self.create_main_menu, font=('Helvetica', 14), bg='#cbcbcb', fg='black')

        self.save_button.pack(pady=20)
        self.back_button.pack(pady=20)



#Algorithm

    def save_data(self):
        learning = self.lr_entry.get()
        epoch = self.epochs_entry.get()
        # math = self.math_entry.get()
        # science = self.science_entry.get()
        # english = self.english_entry.get()
        # pass_fail = self.pass_fail_entry.get().lower()

        try:
            self.learning_rate = float(learning)
            self.max_epochs = int(epoch)
            # math = float(math)
            # science = float(science)
            # english = float(english)
            # if pass_fail not in ['pass', 'fail']:
            #     raise ValueError("Pass/Fail must be either 'pass' or 'fail'")
            #
            # data['Math'].append(math)
            # data['Science'].append(science)
            # data['English'].append(english)
            # data['Pass/Fail'].append(pass_fail)

            # print( "Data saved successfully!")

            # self.math_entry.delete(0, tk.END)
            # self.science_entry.delete(0, tk.END)
            # self.english_entry.delete(0, tk.END)
            # self.pass_fail_entry.delete(0, tk.END)
            self.train_perceptron()
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def train_perceptron(self):
        #Save Data
        learning = self.lr_entry.get()
        epoch = self.epochs_entry.get()
        goal = self.goal_entry.get()
        try:
            self.learning_rate = float(learning)
            self.max_epochs = int(epoch)
            self.goal = float(goal)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        self.mse = [0] * self.max_epochs
        for epoch in range(self.max_epochs):
            self.mse[epoch] = 0
            for i in range(len(data['Math'])):
                X = [data['Math'][i], data['Science'][i], data['English'][i]]
                Y_d = 1 if data['Pass/Fail'][i].lower() == 'pass' else 0
                # Calculate actual output
                weighted_sum = sum(w * x for w, x in zip(self.weights, X)) + self.theta
                weighted_sum = round(weighted_sum, 4)
                Y_a = u(weighted_sum)#Fix python problem
                # Calculate error
                error = Y_d - Y_a
                self.mse[epoch] += pow(error, 2)
                # Update weights
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * X[j]
                    self.weights[j] = round(self.weights[j], 4)
                    # print({j},"\n")
                    # print(self.weights[j])
                # Update threshold
                self.theta += self.learning_rate * error
                self.theta = round(self.theta, 4)
                # Print detailed debug information
                print(f"Epoch: {epoch}, Sample: {i}")
                print(f"Input: {X}, Desired Output: {Y_d}, Actual Output: {Y_a}")
                print(f"Weighted Sum: {weighted_sum}, Error: {error}")
                print(f"Updated Weights: {self.weights}, Updated Theta: {self.theta}")
                print("==============================================")

            self.mse[epoch] /= len(data['Math'])
            if self.mse[epoch] <= self.goal:
                # print(self.goal)
                results_window = tk.Toplevel(self.root)
                results_window.title("Test Results")
                results_window.geometry("400x400")

                results_text = tk.Text(results_window, wrap=tk.WORD)
                results_text.pack(expand=True, fill=tk.BOTH)

                results_text.insert(tk.END, "Epoch # | Mse \n")
                results_text.insert(tk.END, "-" * 60 + "\n")

                for i in range(epoch + 1):
                    result_text = f"Epoch {i + 1:>1}: Mse: {self.mse[i]:<2} \n"
                    results_text.insert(tk.END, result_text)
                break



        messagebox.showinfo("Training Complete!", "Training Complete!")
        self.show_3d_graph()

    def test_perceptron_data(self):
        if not hasattr(self, 'test_data') or not self.test_data['Math']:
            messagebox.showerror("Test Error", "No test data available.")
            return

        actual_results = []
        Desired_restults = []
        mse = 0

        for i in range(len(self.test_data['Math'])):
            X = [self.test_data['Math'][i], self.test_data['Science'][i], self.test_data['English'][i]]
            Y_d = 1 if self.test_data['Pass/Fail'][i] == 'pass' else 0
            weighted_sum = sum(w * x for w, x in zip(self.weights, X)) + self.theta
            weighted_sum = round(weighted_sum, 4)
            Y_a = u(weighted_sum)

            actual_results.append(Y_a)
            Desired_restults.append(Y_d)
            mse += pow((Y_d - Y_a), 2)
            mse = round(mse, 8)

        mse /= len(self.test_data['Math'])
        mse = round(mse, 8)
        results_window = tk.Toplevel(self.root)
        results_window.title("Test Results")
        results_window.geometry("1000x1000")

        results_text = tk.Text(results_window, wrap=tk.WORD)
        results_text.pack(expand=True, fill=tk.BOTH)

        results_text.insert(tk.END, "Sample # | Math | Science | English | Actual | Predicted\n")
        results_text.insert(tk.END, "-" * 60 + "\n")

        for i in range(len(self.test_data['Math'])):
            result_text = (
                f"Sample {i + 1:>2}: "
                f"Math: {self.test_data['Math'][i]:<6} "
                f"Science: {self.test_data['Science'][i]:<7} "
                f"English: {self.test_data['English'][i]:<7} "
                f"Actual: {'Pass' if actual_results[i] == 1 else 'Fail':<5} "
                f"Predicted: {'Pass' if Desired_restults[i] == 1 else 'Fail'}\n"
            )
            results_text.insert(tk.END, result_text)

        mse_text = f"\nMean Squared Error: {mse:.4f}"
        results_text.insert(tk.END, mse_text)

    def test_perceptron_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.math_label = tk.Label(self.root, text="Math:", font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.math_entry = tk.Entry(self.root, font=('Helvetica', 14), bg='#cbcbcb', fg='black')

        self.science_label = tk.Label(self.root, text="Science:", font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.science_entry = tk.Entry(self.root, font=('Helvetica', 14), bg='#cbcbcb', fg='black')

        self.english_label = tk.Label(self.root, text="English:", font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.english_entry = tk.Entry(self.root, font=('Helvetica', 14), bg='#cbcbcb', fg='black')

        self.math_label.pack(pady=5)
        self.math_entry.pack(pady=5)

        self.science_label.pack(pady=5)
        self.science_entry.pack(pady=5)

        self.english_label.pack(pady=5)
        self.english_entry.pack(pady=5)

        self.test_button = tk.Button(self.root, text="Test", command=self.test_perceptron, font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.test_all_button = tk.Button(self.root, text="Test All Data Points", command=self.test_perceptron_data,
                                         font=('Helvetica', 14), bg='#2e2e2e', fg='white')
        self.back_button = tk.Button(self.root, text="Back", command=self.create_main_menu, font=('Helvetica', 14), bg='#cbcbcb', fg='black')

        self.test_button.pack(pady=20)
        self.test_all_button.pack(pady=10)
        self.back_button.pack(pady=20)

        self.math_entry.bind("<Return>", lambda event: self.science_entry.focus_set())
        self.science_entry.bind("<Return>", lambda event: self.english_entry.focus_set())
        self.english_entry.bind("<Return>", lambda event: self.test_perceptron())

    def show_3d_graph(self):
        if not data['Math']:
            messagebox.showerror("Plot Error", "No data available to plot.")
            return

        fig = plt.figure(facecolor='#2e2e2e')
        ax = fig.add_subplot(111, projection='3d', facecolor='#2e2e2e')

        # Prepare data for plotting
        math = data['Math']
        science = data['Science']
        english = data['English']
        pass_fail = data['Pass/Fail']
        colors = ['black' if pf == 'pass' else 'w' for pf in pass_fail]

        # Plot data points
        ax.scatter(math, science, english, c=colors, marker='o')

        # Create a meshgrid for the decision boundary
        x = np.linspace(min(math), max(math), 10)
        y = np.linspace(min(science), max(science), 10)
        X, Y = np.meshgrid(x, y)

        # Calculate Z values for the decision boundary plane
        Z = (-self.theta - self.weights[0] * X - self.weights[1] * Y) / self.weights[2]

        # Plot the decision boundary
        ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, color='cyan')

        # Set axis labels
        ax.set_xlabel('Math', color='white')
        ax.set_ylabel('Science', color='white')
        ax.set_zlabel('English', color='white')
        ax.tick_params(colors='white')

        # Set grid lines
        ax.xaxis._axinfo['grid'].update(color='gray')
        ax.yaxis._axinfo['grid'].update(color='gray')
        ax.zaxis._axinfo['grid'].update(color='gray')
        ax.set_title('Perceptron Boundary Data Graph', color='white')
        plt.show()

    def test_perceptron(self):
        try:
            math = float(self.math_entry.get())
            science = float(self.science_entry.get())
            english = float(self.english_entry.get())

            X = [math, science, english]
            weighted_sum = sum(w * x for w, x in zip(self.weights, X)) + self.theta
            weighted_sum = round(weighted_sum, 4)
            Y_a = u(weighted_sum)
            result = "Pass" if Y_a == 1 else "Fail"
            messagebox.showinfo("Test Result", f"The predicted result is: {result}")

        except ValueError as e:
            print(str(e))
            messagebox.showerror("Error", "Please enter valid numerical values for Math, Science, and English.")



if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()
