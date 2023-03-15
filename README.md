# LDABenchmark
OPTED WP6 Testing

This code is designed to test several hypotheses on trimming and stop word generation. 
It uses gensim LDA and provide a theta matrix. 
A secondary function is calculating optimal k value using gensim's own CoherenceModel function.

Guy Shababo March 15 2023

Functions: 

Main: Runs the main loop
read_corpus_file: Reads the corpus as one file
split_to_sentences: receives text and a list of stopwords and provide nonempty sentences
read_stopwords - Read stopwords from a file 
dump_text_as_file - FFU
flatten - FFU (Flattens a list)
jaccard_similarity FFU
save_to_excel: Saves topics to an Excel file 

List of constants 
STOP_CHARS Token delimiters
CHARS_TO_REMOVE Characters that are not analyzed 
COLLECTION_PATH FFU 
DO_COHERENCE A flag to calculate coherence (best K) and not normal LDA
EXCEL_FILE Output file
STOP_WORD_FILE Path to stop words 
CORPUS_FILE Path to corpus 
