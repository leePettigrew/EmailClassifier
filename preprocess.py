"""
preprocess.py

This module handles all data preprocessing steps for the multi-label email classification project.
It:
  1. Loads input data from CSV files specified in config.
  2. Renames columns to shorter versions (e.g., Type 1 -> y1, Type 2 -> y2, etc.).
  3. Converts text columns to strings and combines "Ticket Summary" and "Interaction content" into one field.
  4. Applies deduplication to remove repeated or boilerplate text from the interaction content.
  5. Removes noise patterns from the text.
  6. Optionally translates texts to English (if needed).

This design adheres to the brief by keeping preprocessing modular and separated from model code.
"""

import pandas as pd
import re
import config


def get_input_data() -> pd.DataFrame:
    """
    Loads data from the CSV files specified in config, renames columns,
    converts text columns to strings, and combines "Ticket Summary" and
    "Interaction content" into a new column "CombinedText".

    :return: Combined DataFrame from both CSV files.
    """
    # Load CSV files from the data folder
    df1 = pd.read_csv(config.DATA_FILE_1, skipinitialspace=True)
    df2 = pd.read_csv(config.DATA_FILE_2, skipinitialspace=True)

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

    # Ensure text columns are of string type
    df[config.TICKET_SUMMARY_COL] = df[config.TICKET_SUMMARY_COL].astype(str)
    df[config.INTERACTION_CONTENT_COL] = df[config.INTERACTION_CONTENT_COL].astype(str)

    # Combine "Ticket Summary" and "Interaction content" into one field "CombinedText"
    df["CombinedText"] = df[config.TICKET_SUMMARY_COL] + " " + df[config.INTERACTION_CONTENT_COL]

    # Drop rows where primary label (y2) is missing or empty
    df = df.loc[(df["y2"] != '') & (~df["y2"].isna())]

    return df


def de_duplication(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate or repetitive sections from the Interaction content
    using Ticket id as the grouping key. The deduplication process:
      - Splits email text using specific patterns,
      - Removes known boilerplate phrases using pre-defined customer support templates,
      - Retains only unique segments per ticket.

    The deduplicated content is written back to the Interaction content column.

    :param data: Input DataFrame
    :return: DataFrame with cleaned interaction content.
    """
    # Create a temporary column for deduplicated content
    data["ic_deduplicated"] = ""

    # Customer support templates for multiple languages
    cu_template = {
        "english": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"
        ],
        "german": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
            r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?"
        ],
        "french": [
            r"L'équipe d'assistance à la clientèle d'Aspiegel\,?",
            r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le siège est à Dublin, en Irlande\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zélande et dans d'autres pays\.?"
        ],
        "spanish": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
            r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislación de Irlanda con su sede en Dublín, Irlanda\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?"
        ],
        "italian": [
            r"Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
            r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?"
        ],
        "portguese": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
            r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova Zelândia e outros países\.?"
        ]
    }

    # Build a single regex pattern for all customer support templates
    cu_pattern = ""
    for lang_templates in cu_template.values():
        for pattern in lang_templates:
            cu_pattern += f"({pattern})|"
    cu_pattern = cu_pattern.rstrip("|")

    # Define patterns for splitting email text
    pattern_1 = r"(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
    pattern_2 = r"(On.{30,60}wrote:)"
    pattern_3 = r"(Re\s?:|RE\s?:)"
    pattern_4 = r"(\*\*\*\*\*\(PERSON\) Support issue submit)"
    pattern_5 = r"(\s?\*\*\*\*\*\(PHONE\))*$"
    split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"

    # Process deduplication for each unique Ticket id
    tickets = data[config.TICKET_ID_COL].value_counts()

    for t in tickets.index:
        ticket_df = data.loc[data[config.TICKET_ID_COL] == t, :]
        ic_set = set()
        ic_deduplicated = []
        for ic in ticket_df[config.INTERACTION_CONTENT_COL]:
            # Split text based on the defined patterns
            ic_parts = re.split(split_pattern, ic)
            ic_parts = [part for part in ic_parts if part is not None]
            # Remove split patterns from each part
            ic_parts = [re.sub(split_pattern, "", part.strip()) for part in ic_parts]
            # Remove customer support boilerplate using the customer template
            ic_parts = [re.sub(cu_pattern, "", part.strip()) for part in ic_parts]

            current_parts = []
            for part in ic_parts:
                if len(part) > 0 and part not in ic_set:
                    ic_set.add(part)
                    current_parts.append(part + "\n")
            ic_deduplicated.append(" ".join(current_parts))
        # Assign the deduplicated content back for this ticket
        data.loc[data[config.TICKET_ID_COL] == t, "ic_deduplicated"] = ic_deduplicated

    # Optionally, export intermediate results for debugging
    data.to_csv('deduplication_output.csv', index=False)

    # Replace original Interaction content with the deduplicated version
    data[config.INTERACTION_CONTENT_COL] = data["ic_deduplicated"]
    data.drop(columns=["ic_deduplicated"], inplace=True)

    return data


def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes defined noise patterns from Ticket Summary and Interaction content.
    Cleans and standardizes the text for better feature extraction.

    :param df: Input DataFrame
    :return: Cleaned DataFrame
    """
    # Define noise patterns for general removal
    noise = r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    df[config.TICKET_SUMMARY_COL] = df[config.TICKET_SUMMARY_COL].str.lower() \
        .replace(noise, " ", regex=True) \
        .replace(r'\s+', ' ', regex=True).str.strip()
    df[config.INTERACTION_CONTENT_COL] = df[config.INTERACTION_CONTENT_COL].str.lower()

    # Additional noise patterns to remove from Interaction content
    noise_patterns = [
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
        r"i will do my very best to assist you"
        r"in order to give you the best solution",
        r"could you please clarify your request with following information:"
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

    for pattern in noise_patterns:
        df[config.INTERACTION_CONTENT_COL] = df[config.INTERACTION_CONTENT_COL].replace(pattern, " ", regex=True)

    df[config.INTERACTION_CONTENT_COL] = df[config.INTERACTION_CONTENT_COL].replace(r'\s+', ' ', regex=True).str.strip()

    # Optionally, filter out rows with insufficient y1 frequency (if needed)
    good_y1 = df["y1"].value_counts()[df["y1"].value_counts() > 10].index
    df = df.loc[df["y1"].isin(good_y1)]

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
