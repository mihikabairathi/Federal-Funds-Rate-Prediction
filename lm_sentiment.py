import csv

class LMSentimentDict:
    def __init__(self, path_to_csv: str, sentiment_fields: list[str]):
        master_dictionary = {}
        with open(path_to_csv) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                master_dictionary[row["Word"].lower()] = row
                line_count += 1
        for key, item in master_dictionary.items():
            item = {item_key.lower(): item_v for item_key, item_v in item.items()}
            item['word'] = item['word'].lower()
            master_dictionary[key] = item
        for key, item in master_dictionary.items():
            for field, value in item.items():
                if type(value) == str and value.isdigit():
                    item[field] = int(value)
                elif type(value) == str:
                    try:
                        item[field] = float(value)
                    except:
                        pass    
            master_dictionary[key] = item
        self.master_dictionary = master_dictionary
        self.sentiment_fields = sentiment_fields
        
    def calculate_sentiment_score(self, doc: list[str]):
        token_count = 0
        sentiment_counts = {k: 0 for k in self.sentiment_fields}
        for token in doc:
            if token in self.master_dictionary:
                token_count += 1
                for sentiment in self.sentiment_fields:
                    sentiment_counts[sentiment] += int(self.master_dictionary[token][sentiment] != 0)
        return {k: v / token_count for k, v in sentiment_counts.items()}