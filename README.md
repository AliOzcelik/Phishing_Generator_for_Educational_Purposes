# Phishing_Generator_for_Educational_Purposes

Using the Interface, you can design the prompt for fine-tuned Llama3.2, then you can generate the desired email. In prompt; length of the email, the tone, scenerio or topic should be provided. Any extra instruction or constraint for the prompt can be provided in "Extra Instructions" part. 
The prompt for the fine-tuned LLM can be previewed in "Prompt Preview" part. 

<img width="662" height="742" alt="image" src="https://github.com/user-attachments/assets/b64aaace-d072-462a-863b-9f4a784b4467" />

After clicking the "Generate" button, the model wil run and generate the desired email
<img width="608" height="426" alt="image" src="https://github.com/user-attachments/assets/526155d0-df92-4568-b381-9808a545cac2" />



At the left-hand side of the page, the model settings can be seen. In the base model path, "meta-llama/Llama-3.2-1B-Instruct" should be written. In the repository page, there is a file named "chekpoint2". After dowloading it, you need to write the path of checkpoint 2 to the "PEFT adapter path" in the model settings. 
**Note that adapter_model.safetensors should also be downloaded and should be added to the checkpoint2 file.**


<img width="258" height="302" alt="image" src="https://github.com/user-attachments/assets/ac2422d9-8eaf-4e39-b0b1-13712547e920" />
