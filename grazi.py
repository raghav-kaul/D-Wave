import numpy as np

class fooditem:
    def __init__(self,calories,total_fat,saturated_fat,carbs,protiens):
        self.calories = calories
        self.total_fat = total_fat*9
        self.saturated_fat = saturated_fat*9
        self.carbs = carbs*4
        self.protiens = protiens*4

# 
food1 = fooditem(
    420,
    28,
    9,
    23,
    8,
)

food2 = fooditem(
    260,
    10,
    4.5,
    39,
    4.5,
)

food3 = fooditem(
    280,
    2,
    0.3,
    53,
    10,
)

drink = fooditem(
    70,
    1,
    0.5,
    12,
    2,
)

Total_kcal = food1.calories + food2.calories + food3.calories + drink.calories

print(f"Total Calories in your Meal = {Total_kcal}kcal")