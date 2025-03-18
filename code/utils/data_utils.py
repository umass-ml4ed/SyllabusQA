from code.utils.utils import clean_str


def get_targets_answers(test_set):
    targets = [clean_str(item["answer"]) for item in test_set]
    
    return targets


def normalize_question_type(question_type):
    
    return question_type.lower().replace("-", " ")


def get_targets_question_types(test_set):
    targets = [normalize_question_type(clean_str(item["question_type"])) for item in test_set]
    
    return targets


def post_process_predictions(predictions_batch, prompts_len, configs):
    preds_question_types = []
    processed_predictions = []
    raw_predictions = []
    reasoning_steps = []
    for prediction, prompt_len in  zip(predictions_batch, prompts_len):
        # Remove prompt
        prediction = prediction[prompt_len:]
        raw_predictions.append(prediction)
        if( configs.add_question_type ):
            prediction_split = prediction.split("\n")
            question_type = normalize_question_type(clean_str(prediction_split[0]))
            preds_question_types.append(clean_str(question_type))
            prediction = prediction_split[1].split("### The answer is:")[1]
        elif( configs.add_reasoning_steps ):
            prediction_split = prediction.split("\n")
            question_type = normalize_question_type(clean_str(prediction_split[0]))
            preds_question_types.append(clean_str(question_type))
            prediction = "\n".join(prediction_split[1:])
            if( "### The answer is:" in prediction ):
                reasoning_step_then_answer = prediction.split("### The answer is:")
                reasoning_step = reasoning_step_then_answer[0]
                prediction = reasoning_step_then_answer[1]
            else:
                reasoning_step = prediction
                prediction = ""
            reasoning_steps.append(reasoning_step)
        # Clean string
        prediction = clean_str(prediction)
        processed_predictions.append(clean_str(prediction))
        
    return processed_predictions, preds_question_types, reasoning_steps, raw_predictions