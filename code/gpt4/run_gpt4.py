import os
import time

from code.utils.utils import clean_str


def run_gpt4(row, assistant, client, configs):
    prompt = row["prompt"], 
    syllabus_name = row["syllabus_name"]

    # Load syllabus pdf file
    filepath = os.path.join(configs.syllabi_dir, configs.syllabi_type, f"{syllabus_name}.{configs.syllabi_type}")
    file = client.files.create(
        file=open(filepath, "rb"),
        purpose="assistants"
        )
    
    # Add syllabus file to assistant for retrieval
    attach_file(assistant, file, client)

    # Create prompt message
    thread = client.beta.threads.create()
    thread_message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"{prompt}",
        )
    
    # Create run and submit
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
        )

    # Retrieve run to check if completed
    delay_time = 0.5
    while(True):
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
            )
        if (run.status == "completed"):
            break
        # Handle not in progress run status
        elif (run.status != "in_progress"):
            print("Handling bad run status...")
            # Delete thread so we don't share any context between prompts
            delete_thread(thread, client)
            # Remove syllabus file from GPT-4 assistant post completion
            detach_file(assistant, client)
            return run_gpt4(row, assistant, client, configs)
        # Wait before checking run status again
        time.sleep(delay_time)
        delay_time = min(delay_time * 2, 10)        
    assert run.status == "completed", "Error: Run not completed"

    # Get completion
    message_list = client.beta.threads.messages.list(
        thread_id=thread.id
        )
    row["citation"] = message_list.data[0].content[0].text.annotations[0].file_citation.quote if( len(message_list.data[0].content[0].text.annotations) > 0 ) else ""
    # Don't clean predicted answer text at run time in case of edge case throwing errors
    row["predicted_answer_raw"] = message_list.data[0].content[0].text.value

    # Delete thread so we don't share any context between prompts
    delete_thread(thread, client)
    # Remove syllabus file from GPT-4 assistant post completion
    detach_file(assistant, client)

    return row


def delete_thread(thread, client):
    response = client.beta.threads.delete(thread.id)
    assert response.deleted, "Error: Thread not deleted"


def detach_file(assistant, client):
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        file_ids=[],
        )
    assert len(assistant.file_ids) == 0, "Error: File not deleted from assistent"


def attach_file(assistant, file, client):
    assert len(assistant.file_ids) == 0, "Error: Unknown file attached to assistant"
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        file_ids=[file.id],
        )
    assert len(assistant.file_ids) == 1, "Error: File not attached"