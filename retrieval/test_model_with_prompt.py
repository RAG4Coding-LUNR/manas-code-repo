prompt = """
def remove_Occ(s,ch): 
    for i in range(len(s)): 
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    for i in range(len(s) - 1,-1,-1):  
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    return s 
# Write a python function to remove all occurrences of a character in a given string.
def remove_Char(s,c) :  
    counts = s.count(c) 
    s = list(s) 
    while counts :  
        s.remove(c) 
        counts -= 1 
    s = '' . join(s)   
    return (s) 
# Write a function to find the last occurrence of a character in a string.
def last_occurence_char(string,char):
 flag = -1
 for i in range(len(string)):
     if(string[i] == char):
         flag = i
 if(flag == -1):
    return None
 else:
    return flag + 1
# Write a function to remove odd characters in a string.
def remove_odd(str1):
 str2 = ''
 for i in range(1, len(str1) + 1):
    if(i % 2 == 0):
        str2 = str2 + str1[i - 1]
 return str2
# Write a function to remove characters from the first string which are present in the second string.
NO_OF_CHARS = 256
def str_to_list(string): 
	temp = [] 
	for x in string: 
		temp.append(x) 
	return temp 
def lst_to_string(List): 
	return ''.join(List) 
def get_char_count_array(string): 
	count = [0] * NO_OF_CHARS 
	for i in string: 
		count[ord(i)] += 1
	return count 
def remove_dirty_chars(string, second_string): 
	count = get_char_count_array(second_string) 
	ip_ind = 0
	res_ind = 0
	temp = '' 
	str_list = str_to_list(string) 
	while ip_ind != len(str_list): 
		temp = str_list[ip_ind] 
		if count[ord(temp)] == 0: 
			str_list[res_ind] = str_list[ip_ind] 
			res_ind += 1
		ip_ind+=1
	return lst_to_string(str_list[0:res_ind]) 
# Write a python function to remove first and last occurrence of a given character from the string.
Test cases:
assert remove_Occ("hello","l") == "heo"
assert remove_Occ("abcda","a") == "bcd"
assert remove_Occ("PHP","P") == "H"
Code: 
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_path = "/home/avisingh/models/codellama-RAG-v3-wo-chat-complete"
#model_path = "/home/avisingh/models/codellama-RAG-v1-complete"
#model_path = "unsloth/codellama-7b"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512
)

result = text_generator(prompt)

print(result[0]['generated_text'])