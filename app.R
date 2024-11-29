library(shiny)
library(randomForest)
library(tidymodels)
library(ggplot2)
library(rsconnect)

#load pre-trained model
final_model<-readRDS("stroke_prediction_model.rds") 

# Define UI ----
ui<-fluidPage(
  titlePanel("Stroke Prediction App"),
  sidebarLayout(
    sidebarPanel(
      h3("Enter Patient data here"),
      
      #make space for user to enter information
      selectInput("gender","Please specify your biological sex:",choices=c("Male","Female")),
      numericInput("age","What is your age?",value=30,min=0,max=120),
      selectInput("hypertension","Do you have a history of hypertension?",choices=c("No"=0,"Yes"=1)),
      selectInput("heart_disease","Do you have a history of any heart diseases?",choices=c("No"=0,"Yes"=1)),
      selectInput("ever_married", "Have you ever married?", choices=c("No","Yes")),
      selectInput("work_type","What is your current employment type?", 
                  choices = c("Private occupation, cannot disclose"="Private","Self-employed","Work for Government"="Govt_job","Housewife/man"="children","Never enter workforce"="Never_worked")),
      selectInput("Residence_type","What is your current residence area?",choices=c("Urban","Rural")),
      numericInput("avg_glucose_level","What is your average Glucose level? (a normal range is between 70-99mg/dL)",value=80,min=0),
      numericInput("bmi", "What is your BMI?",value=25,min=0),
      selectInput("smoking_status","What is your smoking status?", 
                  choices=c("Unknown","Never smoked"="never smoked","Formerly smoked"="formerly smoked","Smokes"="smokes")),
      
      actionButton("predict","Predict Stroke Risk")
    ),
    
    mainPanel(
      h3("Prediction Results"),
      
      verbatimTextOutput("prediction_result"),
      
      plotOutput("density_plot")
    )
  )
)

#define Server Logic ----
server<-function(input,output) {
  #function to preprocess user input
  preprocess_input<-function(input_data) {
    tibble(
      gender=as.factor(input_data$gender),
      age=as.numeric(input_data$age),
      hypertension=as.numeric(input_data$hypertension),
      heart_disease=as.numeric(input_data$heart_disease),
      ever_married=as.factor(input_data$ever_married),
      work_type=as.factor(input_data$work_type),
      Residence_type=as.factor(input_data$Residence_type),
      avg_glucose_level=as.numeric(input_data$avg_glucose_level),
      bmi=as.numeric(input_data$bmi),
      smoking_status=as.factor(input_data$smoking_status)
    )}
  
  #event for "Predict" button
  prediction_event<-eventReactive(input$predict, {
    
    req(input$age,input$avg_glucose_level,input$bmi) #make sure inputs are not NULL
    #create a new data frame with user input
    new_data<-preprocess_input(input)
    
    #predict using the final model
    tryCatch({
      predictions<-predict(final_model,new_data,type="prob")
      return(predictions)
    }, 
    error=function(e) {
      showNotification("Error: Please check your inputs.",type="error")
      NULL  #return NULL if prediction fails
    })
  })
  
  #display the prediction result
  output$prediction_result<-renderPrint({
    preds<-prediction_event()
    if (is.null(preds)) {
      "Prediction failed. Please ensure all inputs are correctly filled."
    } else {
      paste0(
        "Your predicted probability of stroke is ", 
        round(preds$.pred_1*100,2),"%."
      )
    }
  })
  
  # Density plot for predicted probabilities
  output$density_plot <- renderPlot({
    preds <- prediction_event()
    
    if (is.null(preds) || nrow(data.frame(probability=preds$.pred_1)) < 2) {
      
      #simulate data if insufficient for density plot
      simulated_probs<-rbeta(100,shape1=2,shape2=5) #example simulation
      ggplot(data.frame(probability=simulated_probs)) +
        geom_density(aes(x=probability),fill="blue",alpha=0.5) +
        labs(
          title="Predicted Stroke Risk Distribution (Simulated)",
          x="Predicted Probability",
          y="Density"
        ) +
        theme_minimal()
    } else {
      ggplot(data.frame(probability=preds$.pred_1)) +
        geom_density(aes(x=probability),fill="blue",alpha=0.5) +
        labs(
          title="Predicted Stroke Risk Distribution",
          x="Predicted Probability",
          y="Density"
        ) +
        theme_minimal()
    }
  })
}


#finally, a command to run the app
shinyApp(ui=ui,server=server)
