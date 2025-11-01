import pandas as pd

def filter_and_sort_cars(file_path):
    df = pd.read_csv(file_path)

    year = input("Enter car year: ")
    model = input("Enter car model: ")

    filtered = df[(df['year'].astype(str) == year) & (df['model'].str.contains(model, case=False, na=False))]

    if filtered.empty:
        print("No matching cars found.")
        return

    print(f"\nFound {len(filtered)} matching cars.\n")
    print(filtered[['year', 'make', 'model', 'odometer', 'sellingprice']].head())

    choice = input("\nSort results by selling price? (yes/no): ").lower()
    if choice == 'yes':
        sorted_df = filtered.sort_values(by='sellingprice')
        print("\nTop results after sorting:\n")
        print(sorted_df[['year', 'make', 'model', 'odometer', 'sellingprice']].head())

# Run it
filter_and_sort_cars('car.csv')
