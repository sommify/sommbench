from typing import Literal

from pydantic import BaseModel, Field

# English
WTQA_PROMPT_EN = """
Act as an expert sommelier.
Your task is to answer the following multiple-choice question.
Your response MUST be a single letter (A, B, C, or D) and nothing else.

Question: {question}

Options:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Correct Answer (A, B, C, or D):"""

# Slovak
WTQA_PROMPT_SK = """
Si expert someliér.
Tvojou úlohou je odpovedať na nasledujúcu otázku s možnosťou výberu z viacerých odpovedí.
Tvoja odpoveď MUSÍ byť jediné písmeno (A, B, C alebo D) a nič iné.

Otázka: {question}

Možnosti:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Správna odpoveď je (A, B, C alebo D):
"""

# Danish
WTQA_PROMPT_DK = """
Du er en ekspert-sommelier.
Din opgave er at besvare følgende multiple choice-spørgsmål.
Dit svar SKAL være et enkelt bogstav (A, B, C eller D) og intet andet.

Spørgsmål: {question}

Valgmuligheder:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Korrekt svar (A, B, C eller D):
"""

# German
WTQA_PROMPT_DE = """
Verhalten Sie sich wie ein erfahrener Sommelier.
Ihre Aufgabe ist es, die folgende Multiple-Choice-Frage zu beantworten.
Ihre Antwort MUSS aus einem einzigen Buchstaben (A, B, C oder D) bestehen und darf nichts anderes enthalten.

Frage: {question}

Optionen:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Richtige Antwort (A, B, C oder D):
"""

# Italian
WTQA_PROMPT_IT = """
Agisci come un sommelier esperto.
Il tuo compito è rispondere alla seguente domanda a scelta multipla.
La tua risposta DEVE essere una sola lettera (A, B, C o D) e nient'altro.

Domanda: {question}

Opzioni:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Risposta corretta (A, B, C o D):
"""

# Spanish
WTQA_PROMPT_ES = """
Actúa como un sumiller experto.
Tu tarea consiste en responder a la siguiente pregunta de opción múltiple.
Su respuesta DEBE ser una sola letra (A, B, C o D) y nada más.

Pregunta: {question}

Opciones:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Respuesta correcta (A, B, C o D):
"""

# Finnish
WTQA_PROMPT_FI = """
Toimi maailmanluokan sommelierina.
Tehtävänäsi on vastata seuraavaan monivalintakysymykseen.
Vastauksesi on oltava yksi kirjain (A, B, C tai D) eikä mitään muuta.

Kysymys: {question}

Vaihtoehdot:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Oikea vastaus (A, B, C tai D):
"""

# Swedish
WTQA_PROMPT_SV = """
Agera som en expert-sommelier.
Din uppgift är att svara på följande flervalsfråga.
Ditt svar MÅSTE bestå av en enda bokstav (A, B, C eller D) och inget annat.

Fråga: {question}

Alternativ:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Rätt svar (A, B, C eller D):
"""


WTQA_PROMPTS = {
    "en": WTQA_PROMPT_EN,
    "sk": WTQA_PROMPT_SK,
    "da": WTQA_PROMPT_DK,
    "de": WTQA_PROMPT_DE,
    "it": WTQA_PROMPT_IT,
    "es": WTQA_PROMPT_ES,
    "fi": WTQA_PROMPT_FI,
    "sv": WTQA_PROMPT_SV,
}


def build_wtqa_prompt(
    question: str, a: str, b: str, c: str, d: str, lang: str = "en"
) -> str:
    template = WTQA_PROMPTS.get(lang, WTQA_PROMPT_EN)
    return template.format(question=question, a=a, b=b, c=c, d=d).strip()


FWP_PROMPT = """
Act as an expert sommelier.
Your task is to evaluate the pairing of a given wine and recipe.
Your response MUST be Yes or No and nothing else.

Recipe: {recipe}
Wine: {wine}
Does the wine pair well with the recipe? Yes or No:"""


def build_fwp_prompt(recipe: str, wine: str) -> str:
    return FWP_PROMPT.format(recipe=recipe, wine=wine).strip()


EN_WFC_PROMPT = """
Analyze the following wine description:
{passage}

Based on this text, populate ALL fields of the required JSON structure.
For any attributes not explicitly mentioned, predict the most likely value based on the other information provided, ensuring the predicted value adheres to the required data type and enum constraints.
"""


class EN_WineSchema(BaseModel):
    type: Literal["red", "white", "rose", "sparkling", "dessert", "fortified"] = Field(
        description="The color / type of the wine."
    )
    sugar: float = Field(description="The residual sugar level in g/L.")
    alcohol: float = Field(
        description="The alcohol content in percentage (%).",
    )
    country: str = Field(description="The country where the wine was produced.")
    region: str = Field(
        description="The specific geographical region or appellation within the country."
    )
    grapes: list[str] = Field(description="A list of the primary grape varietals used.")
    dryness: Literal["dry", "medium dry", "medium sweet", "sweet"] = Field(
        description="A classification of the wine's perceived sweetness."
    )
    body: Literal["light bodied", "medium bodied", "full bodied"] = Field(
        description="A measure of the wine's weight and mouthfeel."
    )
    acidity: Literal["slightly acidic", "medium acidic", "acidity", "very acidic"] = (
        Field(description="A description of the wine's acid level.")
    )


SK_WFC_PROMPT = """
Na základe nasledujúceho opisu vína:
{passage}

Vyplňte VŠETKY polia požadovanej JSON štruktúry.
V prípade, že niektoré atribúty nie sú v popise výslovne uvedené, doplň najpravdepodobnejšiu hodnotu na základe dostupných informácií. Uistite sa, že predpovedaná hodnota dodržiava požadovaný dátový typ a obmedzenia (napr. povolené hodnoty).
"""


class SK_WineSchema(BaseModel):
    # Field names should stay in English for standardized JSON keys!
    type: Literal[
        "červené", "biele", "ružové", "šumivé", "dezertné", "fortifikované"
    ] = Field(description="Farba / typ vína.")
    sugar: float = Field(description="Zvyškový cukor v g/L.")
    alcohol: float = Field(
        description="Obsah alkoholu v percentách (%).",
    )
    country: str = Field(
        description="Krajina pôvodu, kde bolo víno vyrobené. Odpoved musí byť v angličtine."
    )
    region: str = Field(
        description="Konkrétna geografická oblasť alebo apelácia v rámci krajiny. Odpoveď musí byť v angličtine."
    )
    grapes: list[str] = Field(
        description="Zoznam odrôd hrozna použitých na výrobu. Odpovede musia byť v angličtine."
    )
    dryness: Literal["suché", "polosuché", "polosladké", "sladké"] = Field(
        description="Klasifikácia vnímanej sladkosti vína."
    )
    body: Literal["ľahké", "stredné", "plné"] = Field(
        description="Miera plnosti a pocitu vína v ústach."
    )
    acidity: Literal["mierna", "stredná", "vysoká", "veľmi vysoká"] = Field(
        description="Popis úrovne kyslosti vína."
    )


# Danish
DA_WFC_PROMPT = """
Analyser følgende vinbeskrivelse:
{passage}

Baseret på denne tekst skal du udfylde ALLE felter i den krævede JSON-struktur.
For attributter, der ikke er eksplicit nævnt, skal du forudsige den mest sandsynlige værdi baseret på de øvrige oplysninger, der er angivet, og sikre, at den forudsagte værdi overholder de krævede datatype- og enum-begrænsninger.
"""


class DA_WineSchema(BaseModel):
    type: Literal["rød", "hvid", "rosé", "mousserende", "dessert", "hedvin"] = Field(
        description="Vinens farve / type."
    )
    sugar: float = Field(description="Restsukkerindholdet i g/L.")
    alcohol: float = Field(
        description="Alkoholindholdet i procent (%).",
    )
    country: str = Field(
        description="Landet, hvor vinen er produceret. Svaret skal være på engelsk."
    )
    region: str = Field(
        description="Den specifikke geografiske region eller appellation inden for landet. Svaret skal være på engelsk."
    )
    grapes: list[str] = Field(
        description="En liste over de primære druesorter, der er anvendt. Svaret skal være på engelsk."
    )
    dryness: Literal["tør", "halvtør", "halvsød", "sød"] = Field(
        description="En klassificering af vinens opfattede sødme."
    )
    body: Literal["let fyldig", "medium fyldig", "fuld fyldig"] = Field(
        description="Et mål for vinens vægt og mundfølelse."
    )
    acidity: Literal["let syrlig", "medium syrlig", "syrlig", "meget syrlig"] = Field(
        description="En beskrivelse af vinens syreniveau."
    )


# German
DE_WFC_PROMPT = """
Analysieren Sie die folgende Weinbeschreibung:
{passage}

Füllen Sie auf Grundlage dieses Textes ALLE Felder der erforderlichen JSON-Struktur aus.
Für alle nicht ausdrücklich genannten Attribute sagen Sie den wahrscheinlichsten Wert auf Grundlage der anderen bereitgestellten Informationen voraus und stellen Sie sicher, dass der vorhergesagte Wert den erforderlichen Datentyp- und Enum-Einschränkungen entspricht.
"""


class DE_WineSchema(BaseModel):
    type: Literal["rot", "weiß", "rosé", "schaumwein", "dessertwein", "likörwein"] = (
        Field(description="Die Farbe / Art des Weins.")
    )
    sugar: float = Field(description="Der Restzuckergehalt in g/L.")
    alcohol: float = Field(
        description="Der Alkoholgehalt in Prozent (%).",
    )
    country: str = Field(
        description="Das Herkunftsland des Weins. Die Antwort sollte auf Englisch sein."
    )
    region: str = Field(
        description="Die spezifische geografische Region oder Appellation innerhalb des Landes. Die Antwort sollte auf Englisch sein."
    )
    grapes: list[str] = Field(
        description="Eine Liste der hauptsächlich verwendeten Rebsorten. Die Antwort sollte auf Englisch sein."
    )
    dryness: Literal["trocken", "halbtrocken", "lieblich", "süß"] = Field(
        description="Eine Klassifizierung der wahrgenommenen Süße des Weins."
    )
    body: Literal["leicht", "mittelkräftig", "vollmundig"] = Field(
        description="Ein Maß für das Gewicht und das Mundgefühl des Weins."
    )
    acidity: Literal[
        "leicht säuerlich", "mittlere säure", "säurebetont", "sehr säuerlich"
    ] = Field(description="Eine Beschreibung des Säuregehalts des Weins.")


# Italian
IT_WFC_PROMPT = """
Analizza la seguente descrizione del vino:
{passage}

Sulla base di questo testo, compila TUTTI i campi della struttura JSON richiesta.
Per qualsiasi attributo non menzionato esplicitamente, prevedi il valore più probabile sulla base delle altre informazioni fornite, assicurandoti che il valore previsto rispetti il tipo di dati richiesto e i vincoli di enumerazione.
"""


class IT_WineSchema(BaseModel):
    type: Literal["rosso", "bianco", "rosato", "spumante", "dolce", "fortificato"] = (
        Field(description="Il colore / tipo del vino.")
    )
    sugar: float = Field(description="Il residuo zuccherino in g/L.")
    alcohol: float = Field(
        description="La gradazione alcolica in percentuale (%).",
    )
    country: str = Field(
        description="Il paese in cui è stato prodotto il vino. La risposta dovrebbe essere in inglese."
    )
    region: str = Field(
        description="La specifica regione geografica o denominazione all'interno del paese. La risposta dovrebbe essere in inglese."
    )
    grapes: list[str] = Field(
        description="Un elenco dei principali vitigni utilizzati. La risposta dovrebbe essere in inglese."
    )
    dryness: Literal["secco", "abboccato", "amabile", "dolce"] = Field(
        description="Una classificazione della dolcezza percepita del vino."
    )
    body: Literal["di corpo leggero", "di medio corpo", "di corpo pieno"] = Field(
        description="Una misura del peso e della sensazione in bocca del vino."
    )
    acidity: Literal["poco acido", "media acidità", "acido", "molto acido"] = Field(
        description="Una descrizione del livello di acidità del vino."
    )


# Spanish
ES_WFC_PROMPT = """
Analice la siguiente descripción del vino:
{passage}

Basándose en este texto, rellene TODOS los campos de la estructura JSON requerida.
Para cualquier atributo que no se mencione explícitamente, prediga el valor más probable basándose en el resto de la información proporcionada, asegurándose de que el valor predicho se ajusta al tipo de datos requerido y a las restricciones de enumeración.
"""


class ES_WineSchema(BaseModel):
    type: Literal[
        "tinto", "blanco", "rosado", "espumoso", "de postre", "fortificado"
    ] = Field(description="El color / tipo del vino.")
    sugar: float = Field(description="El nivel de azúcar residual en g/L.")
    alcohol: float = Field(
        description="El contenido de alcohol en porcentaje (%).",
    )
    country: str = Field(
        description="El país donde se produjo el vino. La respuesta debe ser en inglés."
    )
    region: str = Field(
        description="La región geográfica específica o denominación de origen dentro del país. La respuesta debe ser en inglés."
    )
    grapes: list[str] = Field(
        description="Una lista de las principales variedades de uva utilizadas. La respuesta debe ser en inglés."
    )
    dryness: Literal["seco", "semiseco", "semidulce", "dulce"] = Field(
        description="Una clasificación de la dulzura percibida del vino."
    )
    body: Literal["cuerpo ligero", "cuerpo medio", "cuerpo completo"] = Field(
        description="Una medida del peso y la sensación en boca del vino."
    )
    acidity: Literal["ligeramente ácido", "acidez media", "ácido", "muy ácido"] = Field(
        description="Una descripción del nivel de acidez del vino."
    )


# Finnish
FI_WFC_PROMPT = """
Analysoi seuraava viinin kuvaus:
{passage}

Täytä tämän tekstin perusteella KAIKKI vaaditun JSON-rakenteen kentät.
Jos jotakin attribuuttia ei ole nimenomaisesti mainittu, ennusta todennäköisin arvo muiden annettujen tietojen perusteella ja varmista, että ennustettu arvo noudattaa vaadittua tietotyyppiä ja luettelorajoituksia.
"""


class FI_WineSchema(BaseModel):
    type: Literal["punainen", "valkoinen", "rosee", "kuohuva", "makea", "väkevöity"] = (
        Field(description="Viinin väri / tyyppi.")
    )
    sugar: float = Field(description="Jäännössokerin määrä grammoina litrassa (g/L).")
    alcohol: float = Field(
        description="Alkoholipitoisuus prosentteina (%).",
    )
    country: str = Field(
        description="Maa, jossa viini on tuotettu. Vastaus tulee olla englanniksi."
    )
    region: str = Field(
        description="Tarkka maantieteellinen alue tai alkuperäluokitus. Vastaus tulee olla englanniksi."
    )
    grapes: list[str] = Field(
        description="Lista tärkeimmistä käytetyistä rypälelajikkeista. Vastaus tulee olla englanniksi."
    )
    dryness: Literal["kuiva", "puolikuiva", "puolimakea", "makea"] = Field(
        description="Luokitus viinin aistetusta makeudesta."
    )
    body: Literal["kevyt", "keskitäyteläinen", "täyteläinen"] = Field(
        description="Mittari viinin painosta ja suutuntumasta."
    )
    acidity: Literal[
        "vähähappoinen", "keskihappoinen", "happoinen", "erittäin happoinen"
    ] = Field(description="Kuvaus viinin happotasosta.")


# Swedish
SV_WFC_PROMPT = """
Analysera följande vinbeskrivning:
{passage}

Baserat på denna text ska du fylla i ALLA fält i den JSON-struktur som krävs.
För alla attribut som inte uttryckligen nämns, förutsäg det mest sannolika värdet baserat på övrig information som tillhandahålls, och se till att det förutsagda värdet följer de nödvändiga datatyp- och enumbegränsningarna.
"""


class SV_WineSchema(BaseModel):
    type: Literal["rött", "vitt", "rosé", "mousserande", "dessert", "stark"] = Field(
        description="Vinets färg / typ."
    )
    sugar: float = Field(description="Restockerhalten i g/L.")
    alcohol: float = Field(
        description="Alkoholhalten i procent (%).",
    )
    country: str = Field(
        description="Landet där vinet producerades. Svaret ska vara på engelska."
    )
    region: str = Field(
        description="Den specifika geografiska regionen eller appellationen inom landet. Svaret ska vara på engelska."
    )
    grapes: list[str] = Field(
        description="En lista över de primära druvsorterna som används. Svaret ska vara på engelska."
    )
    dryness: Literal["torr", "halvtorr", "halvsöt", "söt"] = Field(
        description="En klassificering av vinets upplevda sötma."
    )
    body: Literal["lätt", "medelfyllig", "fyllig"] = Field(
        description="Ett mått på vinets tyngd och munkänsla."
    )
    acidity: Literal["låg syra", "medelhög syra", "hög syra", "mycket hög syra"] = (
        Field(description="En beskrivning av vinets syranivå.")
    )


WFC_PROMPTS = {
    "en": EN_WFC_PROMPT,
    "sk": SK_WFC_PROMPT,
    "da": DA_WFC_PROMPT,
    "de": DE_WFC_PROMPT,
    "it": IT_WFC_PROMPT,
    "es": ES_WFC_PROMPT,
    "fi": FI_WFC_PROMPT,
    "sv": SV_WFC_PROMPT,
}


def build_wfc_prompt(passage: str, language: str) -> str:
    if language not in WFC_PROMPTS:
        raise ValueError(f"Unsupported language: {language}")
    return WFC_PROMPTS[language].format(passage=passage).strip()


WFC_SCHEMAS: dict[str, type[BaseModel]] = {
    "en": EN_WineSchema,
    "sk": SK_WineSchema,
    "da": DA_WineSchema,
    "de": DE_WineSchema,
    "it": IT_WineSchema,
    "es": ES_WineSchema,
    "fi": FI_WineSchema,
    "sv": SV_WineSchema,
}


def get_wfc_schema(language: str) -> type[BaseModel]:
    if language not in WFC_SCHEMAS:
        raise ValueError(f"Unsupported language: {language}")
    return WFC_SCHEMAS[language]
