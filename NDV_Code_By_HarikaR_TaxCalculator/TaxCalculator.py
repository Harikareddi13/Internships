# Input Section
ctc = float(input("Enter your Total CTC (₹): "))
bonus = float(input("Enter your Bonus Amount (₹): "))
deduction = float(input("Enter total deductions (₹): "))

# Calculation Section
total_income = ctc + bonus
taxable_income = max(total_income - deduction, 0)
taxable_above_250000 = max(taxable_income - 250000, 0)
tax = taxable_above_250000 * 0.05  # Assuming flat 5% above ₹2,50,000

# Output Section
print(f"\nTotal Income (CTC + Bonus): ₹{total_income:.2f}")
print(f"Total Deductions: ₹{deduction:.2f}")
print(f"Taxable Income: ₹{taxable_income:.2f}")
print(f"Tax Payable: ₹{tax:.2f}")
