"""
preprocess.py

  1. Loads input data from CSV files specified in config.
  2. Renames columns to shorter versions (e.g., Type 1 -> y1, Type 2 -> y2, etc.).
  3. Converts text columns to strings and combines "Ticket Summary " and "Interaction content" into one field.
  4. Applies deduplication to remove repeated or boilerplate text from the interaction content
  5. Removes noise patterns from the text.

"""

import pandas as pd
import re
import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_input_data() -> pd.DataFrame:
    """
    Loads data from the CSV files specified in config, renames columns,
    converts text columns to strings, and combines "Ticket Summary" and
    "Interaction content" into a new column "CombinedText".

    :return: Combined DataFrame from both CSV files.
    """
    try:
        df1 = pd.read_csv(config.DATA_FILE_1, skipinitialspace=True)
        df2 = pd.read_csv(config.DATA_FILE_2, skipinitialspace=True)
        logging.info(f"Loaded {len(df1)} records from {config.DATA_FILE_1} and {len(df2)} from {config.DATA_FILE_2}")
    except Exception as e:
        logging.error("Error loading CSV files: %s", e)
        raise

    # Rename label columns to short forms for convenience
    rename_dict = {
        config.TYPE1_COL: "y1",
        config.TYPE2_COL: "y2",
        config.TYPE3_COL: "y3",
        config.TYPE4_COL: "y4"
    }
    df1.rename(columns=rename_dict, inplace=True)
    df2.rename(columns=rename_dict, inplace=True)

    # Concatenate the two dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    logging.info(f"Total records after concatenation: {len(df)}")

    # Ensure text columns are strings
    df[config.TICKET_SUMMARY_COL] = df[config.TICKET_SUMMARY_COL].astype(str)
    df[config.INTERACTION_CONTENT_COL] = df[config.INTERACTION_CONTENT_COL].astype(str)

    # Combine "Ticket Summary" and "Interaction content" into "CombinedText"
    df["CombinedText"] = df[config.TICKET_SUMMARY_COL] + " " + df[config.INTERACTION_CONTENT_COL]
    logging.info("Combined 'Ticket Summary' and 'Interaction content' into 'CombinedText'.")

    # Drop rows where primary label (y2) is missing or empty
    before_drop = len(df)
    df = df.loc[(df["y2"] != '') & (~df["y2"].isna())]
    after_drop = len(df)
    logging.info(f"Dropped {before_drop - after_drop} records due to missing primary label (y2).")

    return df

def de_duplication(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate or repetitive sections from the Interaction content using Ticket id as the grouping key.
    Splits email text using specified patterns and removes boilerplate phrases.
    The deduplicated content is written back to the Interaction content column.

    :param data: Input DataFrame.
    :return: DataFrame with deduplicated interaction content.
    """
    data = data.copy()  # Avoid modifying the original dataframe
    # Create a temporary column for deduplicated content
    data["ic_deduplicated"] = ""

    # Define customer support templates (example using English)
    cu_template = {
        "english": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"
        ]
    }
    # Build a single regex pattern for the templates
    cu_pattern = "|".join(f"({pattern})" for pattern in cu_template["english"])

    # Define splitting patterns (customize these based on your email structure)
    split_patterns = [
        r"(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)",
        r"(On.{30,60}wrote:)",
        r"(Re\s?:|RE\s?:)",
        r"(\*\*\*\*\*\(PERSON\) Support issue submit)",
        r"(\s?\*\*\*\*\*\(PHONE\))*$"
    ]
    split_pattern = "|".join(split_patterns)

    # Process deduplication for each unique Ticket id
    ticket_ids = data[config.TICKET_ID_COL].unique()
    logging.info("Starting deduplication for %d tickets", len(ticket_ids))
    for ticket_id in ticket_ids:
        ticket_df = data[data[config.TICKET_ID_COL] == ticket_id]
        ic_set = set()
        dedup_contents = []
        for ic in ticket_df[config.INTERACTION_CONTENT_COL]:
            # Split text based on defined patterns
            parts = re.split(split_pattern, ic)
            parts = [part for part in parts if part]  # Remove empty parts
            # Clean each part: remove remaining split patterns and boilerplate patterns
            parts = [re.sub(split_pattern, "", part.strip()) for part in parts]
            parts = [re.sub(cu_pattern, "", part.strip()) for part in parts]
            # Retain only unique parts
            unique_parts = []
            for part in parts:
                if part and part not in ic_set:
                    ic_set.add(part)
                    unique_parts.append(part)
            dedup_contents.append(" ".join(unique_parts))
        # Assign deduplicated content for this ticket
        data.loc[data[config.TICKET_ID_COL] == ticket_id, "ic_deduplicated"] = dedup_contents

    # Optionally, save intermediate deduplication output for debugging
    data.to_csv('deduplication_output.csv', index=False)
    # Replace original Interaction content with deduplicated version and drop temporary column
    data[config.INTERACTION_CONTENT_COL] = data["ic_deduplicated"]
    data.drop(columns=["ic_deduplicated"], inplace=True)
    logging.info("Deduplication completed.")
    return data

def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes defined noise patterns from Ticket Summary and Interaction content.
    Cleans and standardizes the text for better feature extraction.

    :param df: Input DataFrame.
    :return: Cleaned DataFrame.
    """
    # Remove noise from Ticket Summary
    summary_noise = r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)"
    df[config.TICKET_SUMMARY_COL] = df[config.TICKET_SUMMARY_COL].str.lower().replace(summary_noise, " ", regex=True)
    df[config.TICKET_SUMMARY_COL] = df[config.TICKET_SUMMARY_COL].replace(r'\s+', ' ', regex=True).str.strip()

    # Remove noise from Interaction content
    interaction_noise = [
        r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"\d{2}(:|.)\d{2}",
        r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        r"dear ((customer)|(user))",
        r"dear",
        r"(hello)|(hallo)|(hi )|(hi there)",
        r"good morning",
        r"thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        r"thank you for contacting us",
        r"thank you for your availability",
        r"thank you for providing us this information",
        r"thank you for contacting",
        r"thank you for reaching us (back)?",
        r"thank you for patience",
        r"thank you for (your)? reply",
        r"thank you for (your)? response",
        r"thank you for (your)? cooperation",
        r"thank you for providing us with more information",
        r"thank you very kindly",
        r"thank you( very much)?",
        r"i would like to follow up on the case you raised on the date",
        r"i will do my very best to assist you",
        r"in order to give you the best solution",
        r"could you please clarify your request with following information:",
        r"in this matter",
        r"we hope you(( are)|('re)) doing ((fine)|(well))",
        r"i would like to follow up on the case you raised on",
        r"we apologize for the inconvenience",
        r"sent from my huawei (cell )?phone",
        r"original message",
        r"customer support team",
        r"(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland\.",
        r"(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        r"canada, australia, new zealand and other countries",
        r"\d+",
        r"[^0-9a-zA-Z]+",
        r"(\s|^).(\s|$)"
    ]
    for pattern in interaction_noise:
        df[config.INTERACTION_CONTENT_COL] = df[config.INTERACTION_CONTENT_COL].replace(pattern, " ", regex=True)
    df[config.INTERACTION_CONTENT_COL] = df[config.INTERACTION_CONTENT_COL].replace(r'\s+', ' ', regex=True).str.strip()

    logging.info("Noise removal completed.")
    return df

def translate_to_en(texts: list[str]) -> list[str]:
    """
    Translates a list of texts to English using language detection via Stanza and translation
    via Hugging Face's M2M100 model. Only translates texts that are not already in English.

    :param texts: List of strings (texts) to translate.
    :return: List of texts in English.
    """
    import stanza
    from stanza.pipeline.core import DownloadMethod
    from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer

    t2t_model_name = "facebook/m2m100_418M"
    t2t_pipe = pipeline(task='text2text-generation', model=t2t_model_name)

    model = M2M100ForConditionalGeneration.from_pretrained(t2t_model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_model_name)

    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                 download_method=DownloadMethod.REUSE_RESOURCES)
    text_en_list = []
    for text in texts:
        if text == "":
            text_en_list.append(text)
            continue

        doc = nlp_stanza(text)
        if doc.lang == "en":
            text_en_list.append(text)
        else:
            lang = doc.lang
            # Map some language codes if needed
            if lang == "fro":  # Old French
                lang = "fr"
            elif lang == "la":  # Latin
                lang = "it"
            elif lang == "nn":  # Norwegian Nynorsk
                lang = "no"
            elif lang == "kmr":  # Kurmanji
                lang = "tr"

            # Use translation (defaulting to case 2)
            tokenizer.src_lang = lang
            encoded_input = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id("en"))
            translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            text_en_list.append(translated_text)
    return text_en_list

if __name__ == "__main__":
    # For testing purposes only
    df = get_input_data()
    df = de_duplication(df)
    df = noise_remover(df)
    df.to_csv("preprocessed_output.csv", index=False)
    logging.info("Preprocessing complete. Output saved to preprocessed_output.csv")