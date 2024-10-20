import base64
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
import os

load_dotenv()

# First example
code = """
import matplotlib.pyplot as plt

# Prepare data
authors = ['Author A', 'Author B', 'Author C', 'Author D']
sales = [100, 200, 300, 400]

# Create and customize the bar char
plt.figure(figsize=(10, 6))
plt.bar(authors, sales, label='Books Sold', color='blue')
plt.xlabel('Authors')
plt.ylabel('Number of Books Sold')
plt.title('Book Sales by Authors')

# Display the chart
plt.tight_layout()
plt.show()
"""

sandbox = Sandbox()
execution = sandbox.run_code(code)
chart = execution.results[0].chart

print('Type:', chart.type)
print('Title:', chart.title)
print('X Label:', chart.x_label)
print('Y Label:', chart.y_label)
print('X Unit:', chart.x_unit)
print('Y Unit:', chart.y_unit)
print('Elements:')
for element in chart.elements:
    print('\n  Label:', element.label)
    print('  Value:', element.value)
    print('  Group:', element.group)
# Second example
code_to_run = """
import matplotlib.pyplot as plt

days = list(range(1, 366))
initial_amount = 1000
rate = 1 / 100
amounts = [initial_amount * (1 + rate) ** day for day in days]

plt.plot(days, amounts)
plt.xlabel('Days')
plt.ylabel('Amount')
plt.title('1 Percent Growth Compounded Daily Over a Year')
plt.show()
"""

sandbox = Sandbox()

# Run the code inside the sandbox
execution = sandbox.run_code(code_to_run)

# There's only one result in this case - the plot displayed with `plt.show()`
first_result = execution.results[0]

if first_result.png:
  # Save the png to a file. The png is in base64 format.
  with open('chart.png', 'wb') as f:
    f.write(base64.b64decode(first_result.png))
  print('Chart saved as chart.png')
