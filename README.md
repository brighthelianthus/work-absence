#Objective
Using historic multi-variate data and predicting target variable values for new data.
Classifying an individual to be excessively absent if absenteism hours is more than 3.

#Dataset info

1. Individual identification (ID)
2. Reason for absence (ICD).
Absences attested by the International Code of Diseases (ICD) stratified into 21 categories (I to XXI) as follows:
Category 0 : No reason stated
Category 1 to 14 : Serious issues
Category 15 to 17 : Pregnancy and childbirth related issues
Category 18 to 21 : Fatal issues
Category 22 to 28 : Light issues.

1 Certain infectious and parasitic diseases
2 Neoplasms
3 Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
4 Endocrine, nutritional and metabolic diseases
5 Mental and behavioural disorders
6 Diseases of the nervous system
7 Diseases of the eye and adnexa
8 Diseases of the ear and mastoid process
9 Diseases of the circulatory system
10 Diseases of the respiratory system
11 Diseases of the digestive system
12 Diseases of the skin and subcutaneous tissue
13 Diseases of the musculoskeletal system and connective tissue
14 Diseases of the genitourinary system
15 Pregnancy, childbirth and the puerperium
16 Certain conditions originating in the perinatal period
17 Congenital malformations, deformations and chromosomal abnormalities
18 Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
19 Injury, poisoning and certain other consequences of external causes
20 External causes of morbidity and mortality
21 Factors influencing health status and contact with health services.

And 7 categories without (CID):
22 patient follow-up , 
23 medical consultation , 
24 blood donation ,
25 laboratory examination , 
26 unjustified absence , 
27 physiotherapy ,
dental consultation.
3. Month of absence
4. Day of the week (Monday (2), Tuesday (3), Wednesday (4), Thursday (5), Friday (6))
5. Seasons (summer (1), autumn (2), winter (3), spring (4))
6. Transportation expense
7. Distance from Residence to Work (kilometers)
8. Service time
9. Age
10. Work load Average/day
11. Hit target
12. Disciplinary failure (yes=1; no=0)
13. Education (high school (1), graduate (2), postgraduate (3), master and doctor (4))
14. Son (number of children)
15. Social drinker (yes=1; no=0)
16. Social smoker (yes=1; no=0)
17. Pet (number of pet)
18. Weight
19. Height
20. Body mass index
21. Absenteeism time in hours (target)

#Feeding new data 
Absenteeism_module.py , absenteeism_scaler_object and linear_reg_model are consumed by Absenteeism_Integration.ipynb and predicted outputs are stored in MySQL table predicted_outputs.
