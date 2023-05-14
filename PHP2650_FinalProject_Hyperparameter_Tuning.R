###############
### Library ###
###############

library(tidyverse)
library(ggplot2)
library(kableExtra)
library(tableone)
library(ggpubr)
library(stringr)
library(scales)
library(tensorflow)
library(keras)
library(tokenizers)
library(readtext)
library(tidytext)
library(textclean)
library(hunspell)
library(furrr)
library(yardstick)
library(stopwords)

#################
### Read Data ###
#################

getData <- function(filepath) {
  con <- file(filepath)
  on.exit(close(con))
  data <-readLines(con)
  return(data)
}

train = data.frame(tweet = getData('/Users/amirahff/Documents/Brown Biostatistics/PHP 2650/FinalProject/train_text.txt'))
train$label = getData('/Users/amirahff/Documents/Brown Biostatistics/PHP 2650/FinalProject/train_labels.txt')

val = data.frame(tweet = getData('/Users/amirahff/Documents/Brown Biostatistics/PHP 2650/FinalProject/val_text.txt'))
val$label = getData('/Users/amirahff/Documents/Brown Biostatistics/PHP 2650/FinalProject/val_labels.txt')

test = data.frame(tweet = getData('/Users/amirahff/Documents/Brown Biostatistics/PHP 2650/FinalProject/test_text.txt'))
test$label = getData('/Users/amirahff/Documents/Brown Biostatistics/PHP 2650/FinalProject/test_labels.txt')

#####################
### Preprocessing ###
#####################

# Function for tweet cleaning
tweet_cleaning <- function(x) {
  x = replace_non_ascii(x) # Remove non ascii character (i.e. emoji)
  x = tolower(x) # Cast all to lowercase
  x = str_replace_all(x, pattern = "\\@.*? |\\@.*?[:punct:]", replacement = " ") # Remove mentions
  x = replace_url(x) # Remove url if any
  x = replace_html(x) # Remove html syntax if any
  x = replace_contraction(x) # Remove contraction (')
  x = replace_word_elongation(x) # Remove exaggerated words and make it normal
  x = str_replace_all(x,"\\?", " questionmark") # Keep ? symbol and replace it as questionmark
  x = str_replace_all(x,"\\!", " exclamationmark") # Keep ! symbol and replace it as exclamationmark
  x = str_replace_all(x,"[:punct:]", " ") # Remove all punctuation
  x = str_replace_all(x,"[:digit:]", " ") # Remove all numbers
  x = str_trim(x) # Remove space at the beginning and end of tweet
  x = str_squish(x) # Remove any whitespace in between
  return(x)
}

# Clean tweet
plan(multisession, workers = 4) # Using 4 CPU cores

train <- train %>% 
  mutate(cleanTweetRaw = tweet_cleaning(tweet)) 

test <- test %>% 
  mutate(cleanTweetRaw = tweet_cleaning(tweet)) 

val <- val %>% 
  mutate(cleanTweetRaw = tweet_cleaning(tweet)) 

# List for stopwords
stopWords = c('a','about','an','are','as','at','be','by','for','from','how','in'
              ,'is','it','of','on','or','that','the','this','to','was','what'
              ,'when','where','who','will','with','the')

# Remove Stopwords
train$cleanTweet = unlist(lapply(train$cleanTweetRaw, function(x) {paste(unlist(strsplit(x, " "))[!(unlist(strsplit(x, " ")) %in% stopWords)], collapse=" ")}))

val$cleanTweet = unlist(lapply(val$cleanTweetRaw, function(x) {paste(unlist(strsplit(x, " "))[!(unlist(strsplit(x, " ")) %in% stopWords)], collapse=" ")}))

test$cleanTweet = unlist(lapply(test$cleanTweetRaw, function(x) {paste(unlist(strsplit(x, " "))[!(unlist(strsplit(x, " ")) %in% stopWords)], collapse=" ")}))

# Get num words
num_words = paste(train$cleanTweet, collapse = " ") %>% 
  str_split(" ") %>% 
  unlist() %>% 
  n_distinct()

# Tokenize
tokenizer <- text_tokenizer(num_words = num_words, oov_token='oov') %>% 
  fit_text_tokenizer(train$cleanTweetRaw)

##################
### Prediction ###
##################

# Function to predict
getPrediction <- function(maxlen
                          ,embed_dim
                          ,l1
                          ,l2
                          ,dropout
                          ,recurrence_dropout
                          ,learning_rate
                          ,epoch
                          ,batch_size
                          ,threshold) {
  
  # Apply texts to sequences
  train_x = texts_to_sequences(tokenizer, train$cleanTweet) %>% 
    pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")
  
  test_x = texts_to_sequences(tokenizer, test$cleanTweet) %>% 
    pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")
  
  val_x = texts_to_sequences(tokenizer, val$cleanTweet) %>% 
    pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")
  
  # Transform the target variable on data train
  train_y <- as.integer(train$label)
  test_y <- as.integer(test$label)
  val_y <- as.integer(val$label)
  
  # Set Random Seed for Initial Weight
  tensorflow::tf$random$set_seed(123)
  
  # Build model architecture
  model <- keras_model_sequential(name = "lstm_model") %>% 
    layer_embedding(name = "input",
                    input_dim = num_words,
                    input_length = maxlen,
                    output_dim = embed_dim
    ) %>% 
    layer_lstm(name = "LSTM",
               units = embed_dim,
               kernel_regularizer = regularizer_l1_l2(l1 = l1, l2 = l2),
               dropout = dropout,
               recurrent_dropout = recurrence_dropout,
               return_sequences = F
    ) %>% 
    layer_dense(name = "Output",
                units = 1,
                activation = "sigmoid"
    )
  
  model %>% 
    compile(optimizer = optimizer_adam(learning_rate = learning_rate),
            metrics = "accuracy",
            loss = "binary_crossentropy"
    )
  
  # Run Model
  train_history <- model %>% 
    fit(x = train_x,
        y = train_y,
        batch_size = batch_size,
        epochs = epoch,
        validation_data=list(x_val= test_x, y_val= test_y),
        callbacks = callback_early_stopping(patience = 3),
        # print progress but don't create graphic
        verbose = 0,
        view_metrics = 0
    )
  
  # Predict val data
  val_test = ifelse(model %>% predict(val_x) > threshold, 1, 0)
  
  # Prep for confusion matrix
  decode <- function(x) as.factor(ifelse(x == 0, "Not irony", "Irony"))
  
  true_class <- decode(val$label)
  pred_class <- factor(ifelse(val_test == 0, "Not irony", "Irony"), levels = levels(true_class))

  # Confusion Matrix
  table("Prediction" = pred_class, "Actual" = true_class)
  
  # Make data frame for metrics
  result = data.frame(maxlen = maxlen,
                      embed_dim = embed_dim,
                      l1 = l1,
                      l2 = l2,
                      dropout = dropout,
                      recurrence_dropout = recurrence_dropout,
                      learning_rate = learning_rate,
                      epoch = epoch,
                      batch_size = batch_size,
                      threshold = threshold,
                      accuracy = accuracy_vec(pred_class, true_class),
                      recall = sens_vec(pred_class, true_class),
                      precision = precision_vec(pred_class, true_class),
                      f1 = f_meas_vec(pred_class, true_class)
  )
  
  return(list(result,model))
}

####################
### Model Tuning ###
####################

# Get prediction 
maxlens = c(100,150)
embeds = c(8,16)
l1s = c(0.05, 0.2)
l2s = c(0.05, 0.2)
dropouts = c(0.05, 0.2)
recurrence_dropouts = c(0.05, 0.2)
learning_rates = c(0.001, 0.01)
epochs = c(10,20)
batch_sizes = c(32,64)
thresholds = c(0.5)

allResult = data.frame()
iter = 0
for(maxlen in maxlens) {
  for (embed in embeds) {
    for (l1 in l1s) {
      for (l2 in l2s) {
        for (dropout in dropouts) {
          for (recurrence_dropout in recurrence_dropouts) {
            for (learning_rate in learning_rates) {
              for (epoch in epochs) {
                for (batch_size in batch_sizes) {
                  for (threshold in thresholds) {
                    iter = iter+1
                    result = getPrediction(maxlen = maxlen
                                           ,embed_dim = embed
                                           ,l1 = l1
                                           ,l2 = l2
                                           ,dropout = dropout
                                           ,recurrence_dropout = recurrence_dropout
                                           ,learning_rate = learning_rate
                                           ,epoch = epoch
                                           ,batch_size = batch_size
                                           ,threshold = threshold)
                    allResult = rbind(allResult, result[[1]])
                    save(allResult,file="/Users/amirahff/Documents/Brown Biostatistics/PHP 2650/FinalProject/allResult2.Rda")
                    modelName = paste0('/Users/amirahff/Documents/Brown Biostatistics/PHP 2650/FinalProject/model/model2_',iter,'.hdf5')
                    save_model_hdf5(result[[2]], modelName)
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


