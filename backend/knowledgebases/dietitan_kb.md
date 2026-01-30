# Dietitian Knowledge Base for AI Food Fabrication

## Required inputs
To estimate calories using the method in this knowledge base, the Dietitian Agent needs
1. sex female or male
2. age in years
3. height in cm
4. weight in kg
5. activity level
6. illness condition or stress or injury status
7. meal type

If key fields are missing, the agent should list missing_fields and avoid inventing numbers.

## Resting metabolic rate
Use the Mifflin St Jeor equation published in 1990.

RMR for male
RMR = (10 x weight_kg) + (6.25 x height_cm) - (5 x age_years) + 5

RMR for female
RMR = (10 x weight_kg) + (6.25 x height_cm) - (5 x age_years) - 161

Notes about performance
MSJ performs well for individuals with and without obesity.
Most estimates can underestimate for obese individuals.

## Activity factor
Choose one activity factor AF.

Confined to bed or chair hospitalized
AF 1.0 to 1.2

Ambulatory or light activity
AF around 1.3

Low activity
AF around 1.5

Active
AF around 1.75

Very active
AF around 1.9

## Injury and stress factors
Choose one stress or injury factor IF.

No illness
IF 1.0

Minor surgery or mild or moderate stress
IF 1.1 to 1.3

Major surgery or major trauma or severe stress
IF 1.5 to 1.7

Wound healing
IF 1.1 to 1.3, up to 1.7

Mild infection
IF 1.3

Sepsis
IF 1.2 to 1.5

Long bone fracture
IF 1.3

Closed head injury
IF 1.3 to 1.5

Cancer
IF 1.1 to 1.45

Fever
IF equals 1.2 times RMR per 1 degree C above 37 C

Burns
0 to 20 percent BSA IF 1.0 to 1.5
20 to 40 percent BSA IF 1.5 to 1.85
More than 40 percent BSA IF 1.85 to 2.05

## Daily energy estimate
Estimate total daily energy as

TotalDailyEnergy = RMR x AF x IF

## Adjust for weight gain or loss goals
Pregnancy
Add 340 kcal per day for second trimester
Add 452 kcal per day for third trimester

Lactation
Add 330 kcal per day for first 6 months
Add 400 kcal per day for second 6 months

Underweight weight gain goal
Add 250 to 1000 kcal per day for a gain of one half to two pounds per week

Overweight weight loss goal
Subtract 250 to 500 kcal per day for a loss of one half to one pound per week

## Meal type allocation
When converting daily calories into a target for a single printed food item, allocate a fraction of daily energy based on meal type.

Snack
about 15 percent of daily energy

Meal light
about 25 percent of daily energy

Meal regular
about 30 percent of daily energy

Meal heavy
about 35 percent of daily energy

## Sugar guidance
If no specific sugar target is given, keep sugar low.
A simple cap is about 10 percent of calories, converted to grams using 4 kcal per gram.

## Size and texture guidance for printing
If texture is not specified, default suggestions
Infant and toddler default to soft
Elder default to soft
Others default to normal

For size, select a grams range that matches meal type
Snack 15 to 70 grams depending on age group
Meal light about 80 to 180 grams
Meal regular about 150 to 300 grams
Meal heavy about 250 to 450 grams

## Eating occasion calorie allocation (CACFP based)

When converting daily calories into a target for a single eating occasion, allocate calories using CACFP recommended distribution.

Use Table 6-3 as default targets because it provides consistent, practical meal targets by age group.

Age group 1 year
Breakfast 20 percent
Lunch 26 percent
Dinner 26 percent
Snack 1 14 percent
Snack 2 14 percent

Age group 2 to 4 years
Breakfast 20 percent
Lunch 26 percent
Dinner 26 percent
Snack 1 14 percent
Snack 2 14 percent

Age group 5 years and older, and adults
Breakfast 22 percent
Lunch 32 percent
Dinner 32 percent
Snack 1 7 percent
Snack 2 7 percent

Implementation note
If meal_type is provided as breakfast, lunch, dinner, snack, use the above CACFP allocation.
If meal_type is provided as snack, meal light, meal regular, meal heavy, and no eating occasion is known, fall back to the heuristic allocation below.

Fallback heuristic allocation when eating occasion is unknown
Snack about 15 percent of daily energy
Meal light about 25 percent of daily energy
Meal regular about 30 percent of daily energy
Meal heavy about 35 percent of daily energy

## Macronutrient distribution (AMDR)

After determining target calories for the eating occasion, compute macronutrient targets using AMDR.

Adults and most age groups
Carbohydrate 45 to 65 percent of kcal
Protein 10 to 35 percent of kcal
Fat 20 to 35 percent of kcal

Default macro split for menu planning
If user has no macro constraints, use a practical default split:
Carbohydrate 55 percent
Protein 20 percent
Fat 25 percent

Conversion from kcal to grams
Carbohydrate grams = (kcal * carb_percent) / 4
Protein grams = (kcal * protein_percent) / 4
Fat grams = (kcal * fat_percent) / 9

Fat quality guardrails aligned with Dietary Guidelines
Keep saturated fat under 10 percent of total kcal when possible.
Keep trans fat as low as possible.
