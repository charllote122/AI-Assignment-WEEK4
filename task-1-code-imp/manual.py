import pandas as pd

# Load the car dataset
df = pd.read_csv('car.csv')

# Convert the DataFrame to a list of dictionaries
data = df.to_dict(orient='records')

def sort_by_key(data, key):
    """
    Sort a list of dictionaries (car) by a specified key.
    Example: sort by 'sellingprice' or 'odometer'
    """
    try:
        return sorted(data, key=lambda x: x[key])
    except KeyError:
        print(f"Key '{key}' not found in dataset.")
        return data

# --- Fancy Interactive Part ---
print("Welcome to the Car Data Explorer 🚘")
print("You can view cars by year and model.\n")

# Prompt the user for input
user_year = input("Enter car year (e.g., 2015): ")
user_model = input("Enter car model (e.g., Corolla): ")

# Filter dataset based on input
filtered_cars = [car for car in data if str(car['year']) == user_year and user_model.lower() in str(car['model']).lower()]

if filtered_cars:
    print(f"\n✅ Found {len(filtered_cars)} matching car(s):\n")
    for car in filtered_cars[:5]:
        print(f"{car['year']} {car['make']} {car['model']} | "
              f"Condition: {car['condition']} | "
              f"Odometer: {car['odometer']} | "
              f"Price: ${car['sellingprice']}")
else:
    print("\n⚠️ No matching cars found. Try another year or model.")

# Optionally, sort the filtered list by selling price
if filtered_cars:
    sort_choice = input("\nWould you like to sort these results by price? (yes/no): ").lower()
    if sort_choice == 'yes':
        sorted_cars = sort_by_key(filtered_cars, 'sellingprice')
        print("\n💰 Sorted by Selling Price (Ascending):\n")
        for car in sorted_cars[:5]:
            print(f"{car['year']} {car['make']} {car['model']} - ${car['sellingprice']}")
