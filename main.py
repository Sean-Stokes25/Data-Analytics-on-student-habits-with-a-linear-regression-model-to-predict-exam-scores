import pandas as pd
import pprint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression   #import as sckikit-learn in shell
data = pd.read_csv(r"student_habits_performance.csv")  #import dataset

df = pd.DataFrame(data)

cleaned = pd.DataFrame(df.drop(["student_id","age","gender"],axis = 1)) #remove unwanted columns
#Dictionaries used to map string values to integer values
diet_score_map = {
    "Poor":3,
    "Fair":5,
    "Good":7
    }
parental_education_map = {
    "Noed":0,
    "High School":1,
    "Bachelor":2,
    "Master":3
    }
internet_quality_map = {
    "Poor":0,
    "Average":1,
    "Good":2
    }
part_time_job_map = {
    "No":0,
    "Yes":1
    }
extracurricular_participation_map = {
    "No":0,
    "Yes":1
    }


while True:
    #prompt user
    user_choice = int(input("""\nWhat information would u like to see
    Enter 1 for average student data
    Enter 2 for the affect of parental education on exam results
    Enter 3 for the affect of Wellbeing on exam results
    Enter 4 for the affect a parttime job has on attendence
    Enter 5 for the affect social media has on exam results
    Enter 6 to predict your own exam score\n
    """))
    if user_choice == 1:
        #-----------------
        #Average student
        #----------------
        #Creates the average student
        average_of_data = {
           "Hours of study per day":f"{cleaned['study_hours_per_day'].mean():.2f}",
           "Social media hours per day":f"{cleaned['social_media_hours'].mean():.2f} ",
           "Netflix hours per day":f"{cleaned['netflix_hours'].mean():.2f}",
           "Part time job":cleaned["part_time_job"].mode(),
           "Attendence":f"{cleaned['attendance_percentage'].mean():.2f}",
           "Hours of sleep":f"{cleaned['sleep_hours'].mean():.2f}",
           "Diet quality":cleaned['diet_quality'].mode(),
           "Exercise frequency":f"{cleaned['exercise_frequency'].mean():.2f}",
           "Parental education level":cleaned['parental_education_level'].mode(),
           "Internet quality":cleaned["internet_quality"].mode(),
           "Mental health score":f"{cleaned['mental_health_rating'].mean():.2f}",
           "Extracurriculars":cleaned['extracurricular_participation'].mode(),
           "Exam score":f"{cleaned['exam_score'].mean():.2f}"
           }

        average_student = pd.DataFrame(average_of_data)
        
        #used to display the full dataset not cropped
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(average_student)

    elif user_choice == 2:
        #-------------------------
        #Parental education level
        #-------------------------
        print("Average Exam results  of students sorted by parental education")
        
        #seperates test scores based on parental education then prints the mean of each 
        
        master = cleaned[cleaned["parental_education_level"] == "Master"]
        print(f'Master:{master["exam_score"].mean():.1f}')
         
        highschool = cleaned[cleaned["parental_education_level"] == "High School"]
        print(f'High School:{highschool["exam_score"].mean():.1f}')

        bachelor = cleaned[cleaned["parental_education_level"] == "Bachelor"]
        print(f'Bachelor:{bachelor["exam_score"].mean():.1f}')

        noed = cleaned[cleaned["parental_education_level"] == "Noed"]
        print(f'No education:{noed["exam_score"].mean():.1f}\n')


    elif user_choice == 3:
        #------------------
        #Wellbeing 
        #------------------
        #The wellbeing score is calculated by adding sleep hours mental health rating and diet quality we map diet quality 
        #with the dictionery at the top
        
        
        cleaned["diet_quality"] = cleaned["diet_quality"].map(diet_score_map).fillna(0)  #Maps diet quality

        cleaned["wellbeing_score"] = cleaned["sleep_hours"] + cleaned["mental_health_rating"] + cleaned["diet_quality"] 

        #Creates scatter plot of student wellbeings relationship with examscores 
        x1 = cleaned["wellbeing_score"]
        y1 = cleaned["exam_score"]
        plt.scatter(x1, y1,c=x1,cmap="viridis")   #viridis is the colour scheme
        plt.title("Relationship of wellbeing to exam results")
        plt.colorbar(label="Color scale")
        plt.xlabel("Wellbeing Score")
        plt.ylabel("Exam Score")
        
        plt.show()
    elif user_choice == 4:
        #-----------------------
        #Attendence factors 
        #-----------------------
        #finds mean student attendence of students with and without parttime job
        
        no = cleaned[cleaned["part_time_job"] == "No"]
        print(f'Attendence with parttime job:{no["attendance_percentage"].mean():.1f}')

        yes = cleaned[cleaned["part_time_job"] == "Yes"]
        print(f'Attendence without parttime job:{yes["attendance_percentage"].mean():.1f}')

        plt.show()
    
    elif user_choice == 5:
        #------------------- 
        # Social Media  
        #------------------- 

        df = pd.read_csv(r'student_habits_performance.csv') 

        cleaned = pd.DataFrame(df. drop(["student_id","age","gender"],axis = 1)) 


        cleaned['total_screen_time'] = cleaned['social_media_hours'] + cleaned['netflix_hours'] 

        print(cleaned['total_screen_time'] )
        #gets the average of the soial 

        avg_social_media_hours = cleaned['social_media_hours'].mean() 

        print(f'\nAverage Social Media Hours: {avg_social_media_hours:.1f}') 

        avg_netflix_hours = cleaned['netflix_hours'].mean() 

        print(f'Average Netfix Hours: {avg_netflix_hours:.1f}') 

        #Making the scatter plot to compare exam score and screen time  

        x2= cleaned['total_screen_time'] 

        y2 = cleaned['exam_score'] 

        plt.scatter(x2, y2) 

        plt.title("Exam and Screen Time Comparison") 

        plt.xlabel("Total Screen Time") 

        plt.ylabel("Exam Scores") 

        plt.scatter (x2, y2, c= x2, cmap = 'viridis') 

        plt.show()
    
    
    elif user_choice == 6:
        #Map to relevant dictionaries
        cleaned_for_model = cleaned
        cleaned_for_model["diet_quality"] = cleaned["diet_quality"].map(diet_score_map).fillna(0)
        cleaned_for_model["parental_education_level"] = cleaned["parental_education_level"].map(parental_education_map).fillna(0)
        cleaned_for_model["internet_quality"] = cleaned["internet_quality"].map(internet_quality_map).fillna(0)
        cleaned_for_model["part_time_job"] = cleaned["part_time_job"].map(part_time_job_map).fillna(0)
        cleaned_for_model["extracurricular_participation"] = cleaned["extracurricular_participation"].map(extracurricular_participation_map).fillna(0)
        
        #data that model is trained with
        X = cleaned_for_model[[
            "study_hours_per_day",
            "social_media_hours",
            "netflix_hours",
            "part_time_job",
            "attendance_percentage",
            "sleep_hours",
            "diet_quality",
            "exercise_frequency",
            "parental_education_level",
            "internet_quality",
            "mental_health_rating",
            "extracurricular_participation"]]
        #what data model is trying to predict
        Y = cleaned_for_model["exam_score"]


        model = LinearRegression()   #this creates model

        model.fit(X, Y)  #this train model
        
        #prompts user for their data and predicts their exam score
        predicted_score = model.predict([[
            study := float(input("How many hours study do you do a night:")),
            social_media := float(input("How many hours do u spend on social media a night:")),
            netflix := float(input("How many hours do u watch netflix a day:")),
            job := float(input("Do u have a part time job (enter Yes as 1 or No as 0):")),
            attendence := float(input("What is your attendence percentage:")),
            sleep := float(input("How many hours sleep do u get a night:")),
            diet := float(input("Rate ur diet from 0-2 (0 poor 2 great)")),
            exercise := float(input("How many hours of excercise do u get a day")),
            parental_eeducation := float(input("Enter parents education level(0 = none , 1 = highschool , 2 = bachelor , 3 = master)")),
            internet := float(input("Rate ur internet from 0-2 (0 poor 2 great)")),
            mental_health := float(input("Rate ur mental health from 1-10 (1 poor 10 great)")),
            extracuricular := float(input("Do u participate in extracuricular activity (enter Yes as 1 or No as 0)"))
            ]])
        print(f"Your predicted exam result is : {predicted_score}")
    
    else:
        print("Invalid")
