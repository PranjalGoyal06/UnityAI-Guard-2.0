import re
from datasets import load_dataset
import json
from tqdm import tqdm

finalAns = ""
CATEGORY_KEYWORDS = {
    # --- S1: Violent Crimes ---
    'S1': [
        r'\bassault\w*', r'\battack\w*', r'\babduct\w*', r'\barson\b', r'\bbattery\b',
        r'\bbomb\w*', r'\bextort\w*', r'\bhomicid\w*', r'\bhostage\w*', r'\binjur\w*',
        r'\bkill\w*', r'\bkidnap\w*', r'\bmanslaughter\b', r'\bmurder\w*',
        r'\bphysical\s+abuse\w*', r'\bphysical\s+harm\w*', r'\brob\w*',
        r'\bshoot\w*', r'\bstab\w*', r'\bterroris\w*', r'\btortur\w*', r'\bviolen\w*',
        r'\bdanger\b', r'\bharmful\s+behavior\w*', r'\bgunshot\w*', r'\bgunfire\b',
        r'\bfirearm\w*', r'\bbullet\w*', r'\bblast\w*', r'\bexplosion\w*',
        r'\bfight\w*', r'\bbrawl\w*', r'\bblood\w*', r'\bknife\w*', r'\bhammer\w*',
        r'\bweapon\w*', r'\bgun\w*', r'\brifle\w*', r'\bthreat\w*', r'\bchoke\w*',
        r'\bstrangle\w*', r'\bsuffocate\w*', r'\bhang\w*', r'\bexecut\w*',
        r'\blynch\w*', r'\brape\w*', r'\bgangrape\w*', r'\bmassacre\w*',
        r'\bslaughter\w*', r'\bwar\b', r'\briot\w*', r'\bgrenade\w*', r'\bmachete\w*',
        r'\bcarnage\b', r'\bshootout\w*', r'\bopen\s+fire\w*', r'\bacid\s+attack\w*',
        r'\bthug\w*', r'\bcriminal\w*', r'\bexplosive\w*', r'\bgunman\w*',
        r'\bsniper\w*', r'\bsuicide\s+bomber\w*', r'\bdead\b', r'\bbody\b',
        r'\bcorpse\b', r'\bmangled\b'
    ],

    # --- S2: Non-Violent Crimes ---
    'S2': [
        r'\bbrib\w*', r'\bburglar\w*', r'\bcorrupt\w*', r'\bcounterfeit\w*',
        r'\bdrug\s+deal\w*', r'\bdrug\s+traffic\w*', r'\bembezzle\w*', r'\bforger\w*',
        r'\bfraud\w*', r'\binsider\s+trad\w*', r'\blarceny\b', r'\blaunder\w*',
        r'\bperjur\w*', r'\bscam\w*', r'\bshoplift\w*', r'\bsteal\w*',
        r'\bswindl\w*', r'\btax\s+evasion\w*', r'\btheft\b', r'\bvandalism\b',
        r'\billegal\s+activit\w*', r'\bfinancial\s+loss\w*',
        r'\bavoid\s+detection\b', r'\bconsciousness\s+of\s+guilt\b',
        r'\bavoid\s+accountability\b', r'\bcheat\w*', r'\bwhite[-\s]?collar\w*',
        r'\bscandal\w*', r'\billegal\s+deal\w*', r'\bsmuggle\w*', r'\bblackmail\w*',
        r'\bcyber\s+fraud\w*', r'\bcybercrime\w*', r'\bfake\s+invoice\w*',
        r'\bponzi\b', r'\binvestment\s+fraud\w*', r'\bidentity\s+theft\w*',
        r'\bloan\s+scam\w*', r'\bphishing\w*', r'\bcrypto\s+scam\w*',
        r'\bunauthorized\s+transaction\w*', r'\bmoney\s+scam\w*', r'\bthief\b'
    ],

    # --- S3: Sex Crimes ---
    'S3': [
        r'\bfondl\w*', r'\bgrop\w*', r'\bindecent\s+exposure\w*', r'\bmolest\w*',
        r'\bnon-consensual\w*', r'\brap\w*', r'\bsexual\s+assault\w*',
        r'\bsexual\s+abuse\w*', r'\bsexual\s+harass\w*', r'\bsexual\s+misconduct\w*',
        r'\bsex\s+traffic\w*', r'\bstatutory\s+rape\b', r'\brapist\w*',
        r'\bsexual\w*', r'\bharass\w*', r'\bexpose\w*', r'\bflashing\b',
        r'\bvoyeur\w*', r'\bunwanted\w*', r'\bsexual\s+violence\w*',
        r'\bsex\s+crime\w*', r'\bdate\s+rape\w*', r'\bdrugged\b', r'\bconsent\b',
        r'\bcoerce\w*', r'\binappropriate\s+touch\w*', r'\binappropriate\s+message\w*',
        r'\bsexual\s+predator\w*', r'\bsex\s+offender\w*'
    ],

    # --- S4: Child Exploitation ---
    'S4': [
        r'\bage\s+of\s+consent', r'\bchild\s+abuse\w*', r'\bchild\s+exploit\w*',
        r'\bchild\s+porn\w*', r'\bchild\s+sexual\s+abuse\w*', r'\bCSAM\b',
        r'\bgroom\w*', r'\blur\w*', r'\bminor\w*', r'\bpedophil\w*', r'\bunderage\w*',
        r'\bchild\b', r'\bchildren\b', r'\bkid\w*', r'\bteen\w*', r'\binfant\w*',
        r'\btoddler\w*', r'\bvictim\w*', r'\bpedo\b', r'\bporn\w*', r'\bincest\b'
    ],

    # --- S5: Defamation ---
    'S5': [
        r'\bcharacter\s+assassination\w*', r'\bdefam\w*', r'\bdisparag\w*',
        r'\bfalse\s+accusation\w*', r'\bfalse\s+statement\w*', r'\blibel\w*',
        r'\bslander\w*', r'\bruin\w*\s+reputation\w*', r'\bfalse\s+allegation\w*',
        r'\bfake\s+allegation\w*', r'\bdiscredit\w*', r'\bsmear\w*',
        r'\bmalicious\s+rumor\w*', r'\breputation\w*', r'\brumor\w*',
        r'\bgossip\w*', r'\bbackbite\w*', r'\bspread\s+lies\w*',
        r'\bfalse\s+news\w*', r'\bfalse\s+report\w*'
    ],

    # --- S6: Specialized Advice ---
    'S6': [
        r'\bdiagnos\w*', r'\bprescrib\w*', r'\btreat\w*', r'\bcure\w*',
        r'\bmedical\s+advice\w*', r'\bhealth\s+advice\w*', r'\bfinancial\s+advice\w*',
        r'\binvest\w*\s+advice\w*', r'\binvest\w*\s+recommendation\w*',
        r'\bstock\s+tip\w*', r'\bstock\s+pick\w*', r'\bguaranteed\s+return\w*',
        r'\bbuy\s+this\s+stock\w*', r'\bbuy\s+this\s+crypto\w*',
        r'\blegal\s+advice\w*', r'\blegal\s+opinion\w*',
        r'\bgrounds\s+for\s+lawsuit\w*', r'\badvise\w*\s+you\s+to\s+sue\w*',
        r'\bhow\s+to\s+file\w*\s+a\s+claim\w*', r'\bmedical\b', r'\bhealth\b',
        r'\bmental\s+health\w*', r'\bdoctor\w*', r'\btherapy\w*',
        r'\bpsycholog\w*', r'\bpsychiatrist\w*', r'\bmedication\w*',
        r'\bdrug\w*', r'\bdisease\w*', r'\billness\w*', r'\bcancer\w*',
        r'\bcovid\w*', r'\bflu\b', r'\bvirus\w*', r'\binfection\w*',
        r'\bsymptom\w*', r'\bremedy\b', r'\bfinancial\b', r'\binvestment\w*',
        r'\bstock\w*', r'\bcrypto\w*', r'\bbitcoin\w*', r'\bforex\b',
        r'\btrading\w*', r'\blegal\b', r'\blawyer\w*', r'\battorney\w*',
        r'\bsue\b', r'\bcourt\b', r'\bcase\b', r'\bdivorce\w*',
        r'\bmarriage\s+law\w*', r'\btax\w*', r'\bfiling\w*', r'\baccountant\w*',
        r'\bloan\w*', r'\bmutual\s+fund\w*', r'\binsurance\w*', r'\bpolicy\w*',
        r'\bconsultation\w*', r'\bself[-\s]?diagnosis\w*'
    ],

    # --- S7: Privacy ---
    'S7': [
        r'\bbank\s+account\s+number\w*', r'\bconfidential\s+info\w*',
        r'\bcredit\s+card\s+num\w*', r'\bdata\s+breach\w*', r'\bdox\w*',
        r'\bhome\s+address\w*', r'\bnon-public\s+info\w*', r'\bpassword\w*',
        r'\bPII\b', r'\bpersonal\s+info\w*',
        r'\bpersonally\s+identifiable\s+information\w*', r'\bphone\s+number\w*',
        r'\bphish\w*', r'\bprivacy\s+violation\w*', r'\bprivate\s+data\w*',
        r'\bsocial\s+security\s+number\w*', r'\bSSN\b', r'\bsurveillance\w*',
        r'\bprivacy\b', r'\bprivate\b', r'\bconfidential\b', r'\bcredit\s+card\b',
        r'\bbank\s+account\b', r'\baccount\s+number\w*', r'\bemail\s+address\w*',
        r'\bIP\s+address\w*', r'\bsocial\s+security\b', r'\baadhaar\b',
        r'\bpassport\w*', r'\bdriver\s+license\w*', r'\bleak\w*',
        r'\btracking\w*', r'\bstalker\w*', r'\bcyber\s+stalking\w*',
        r'\bcamera\s+recording\w*'
    ],

    # --- S8: Intellectual Property ---
    'S8': [
        r'\bcopyright\w*', r'\bcounterfeit\w*', r'\bIP\s+theft\w*',
        r'\bintellectual\s+property\w*', r'\bpatent\w*', r'\bpiracy\b',
        r'\bpirat\w*', r'\bplagiaris\w*', r'\bproprietary\s+info\w*',
        r'\btrade\s+secret\w*', r'\btrademark\w*',
        r'\bunauthoriz\w*\s+reproduction\w*', r'\bunauthoriz\w*\s+use\w*',
        r'\bfake\s+product\w*', r'\bknockoff\w*', r'\bunlicensed\w*',
        r'\bcopy\s+content\w*', r'\bmusic\s+piracy\w*', r'\bmovie\s+leak\w*',
        r'\bsoftware\s+crack\w*', r'\btorrent\w*', r'\bdownload\s+illegal\w*',
        r'\billegal\s+stream\w*'
    ],

    # --- S9: Indiscriminate Weapons ---
    'S9': [
        r'\banthrax\b', r'\batomic\s+bomb\w*', r'\bbiological\s+weapon\w*',
        r'\bbioweapon\w*', r'\bchemical\s+weapon\w*', r'\bcluster\s+bomb\w*',
        r'\bcluster\s+munition\w*', r'\bdirty\s+bomb\w*', r'\blandmine\w*',
        r'\bmustard\s+gas\w*', r'\bnerve\s+agent\w*', r'\bnuclear\s+weapon\w*',
        r'\bnuke\w*', r'\bsarin\b', r'\bWMD\b',
        r'\bweapon\w*\s+of\s+mass\s+destruction\w*', r'\bnuclear\b',
        r'\batomic\b', r'\bhydrogen\s+bomb\w*', r'\bgas\s+attack\w*',
        r'\bmissile\w*', r'\brocket\w*', r'\bwarhead\w*', r'\bdrone\s+strike\w*',
        r'\bairstrike\w*', r'\bchemical\s+attack\w*', r'\bmass\s+casualty\w*'
    ],

    # --- S10: Hate ---
    'S10': [
        r'\bantisemit\w*', r'\bbigot\w*', r'\bderogatory\w*', r'\bdiscriminat\w*',
        r'\bhate\s+speech\w*', r'\bhomophob\w*', r'\bislamophob\w*', r'\bKKK\b',
        r'\bneo-nazi\w*', r'\bprejudic\w*', r'\bracial\s+slur\w*',
        r'\bracis\w*', r'\bsexis\w*', r'\bsupremacist\w*', r'\btransphob\w*',
        r'\bxenophob\w*', r'\bdisrespectful\b', r'\bstereotype\w*',
        r'\bnegative\s+sentiment\w*', r'\bappearance\b', r'\bbackground\b',
        r'\breligion\w*', r'\breligious\s+freedom\b', r'\bdiversity\b',
        r'\bdignity\b', r'\bcertain\s+groups\b', r'\bhate\b',
        r'\bhate\s+crime\w*', r'\boffensive\s+comment\w*', r'\bintolerant\b',
        r'\bethnic\s+hatred\w*', r'\bwhite\s+supremacy\w*', r'\bhate\s+group\w*',
        r'\bincite\s+violence\w*'
    ],

    # --- S11: Self-Harm ---
    'S11': [
        r'\banorexi\w*', r'\bbulimi\w*', r'\bcutt\w*\s+myself',
        r'\beating\s+disorder\w*', r'\bend\w*\s+my\s+life\w*', r'\bkill\w*\s+myself',
        r'\boverdos\w*', r'\bpro-ana\b', r'\bpro-mia\b', r'\bself-harm\w*',
        r'\bself\s+harm\w*', r'\bself\s+injur\w*', r'\bself\s+mutilation\w*',
        r'\bsuicid\w*', r'\bwant\w*\s+to\s+die\w*', r'\bdepress\w*',
        r'\bhopeless\b', r'\bworthless\b', r'\blife\s+is\s+pointless\b',
        r'\bhang\s+myself\b', r'\bjump\s+off\b', r'\bbridge\b',
        r'\bburn\s+myself\b', r'\bstarve\w*', r'\bself\s+destruct\w*',
        r'\bhurt\s+myself\b'
    ],

    # --- S12: Sexual Content ---
    'S12': [
        r'\badult\s+film\w*', r'\bBDSM\b', r'\berotic\w*', r'\bexplicit\w*',
        r'\bfetish\w*', r'\bintercourse\b', r'\bkink\w*', r'\blewd\b',
        r'\bnaked\b', r'\bnude\b', r'\borgasm\w*', r'\bpenis\b', r'\bporn\w*',
        r'\bsex\b', r'\bsexual\s+intercourse\w*', r'\bsexually\s+explicit\w*',
        r'\bvagina\b', r'\bXXX\b', r'\bporno\b', r'\bsexual\b', r'\bmasturbat\b',
        r'\barousal\b', r'\bonlyfans\b', r'\bcamgirl\b', r'\bcam\s+show\w*',
        r'\bsexual\s+content\w*', r'\bsex\s+video\w*', r'\bsex\s+tape\w*',
        r'\bstrip\w*', r'\bdirty\s+talk\w*', r'\berotica\b', r'\bgenitals\w*',
        r'\bboobs\b', r'\bbreasts\b', r'\bcum\b', r'\bejaculate\w*',
        r'\bhorny\b', r'\blust\b', r'\bmake\s+love\b', r'\bforeplay\b'
    ],

    # --- S13: Elections --- (Renamed from SS13)
    'S13': [
        r'\bballot\w*', r'\bcandidate\w*', r'\belect\w*', r'\belection\s+fraud\w*',
        r'\belection\s+interference\w*', r'\belection\s+misinformation\w*',
        r'\belection\s+disinformation\w*', r'\bpolitical\s+party\w*', r'\bpoll\w*',
        r'\brigged\s+election\w*', r'\bvote\w*', r'\bvoter\w*',
        r'\bvoter\s+fraud\w*', r'\bvoter\s+suppression\w*', r'\bvoting\w*',
        r'\brigged\b', r'\brigging\b', r'\bfake\s+votes\w*',
        r'\bvote\s+bank\w*', r'\bvote\s+tampering\w*',
        r'\bballot\s+stuffing\w*', r'\bmanifesto\w*', r'\bopposition\w*',
        r'\bgovernment\w*', r'\bminister\w*', r'\bpresident\w*',
        r'\bprime\s+minister\w*', r'\bMP\b', r'\bMLA\b', r'\bparliament\w*',
        r'\bcampaign\w*', r'\bpropaganda\b', r'\belection\s+day\w*'
    ],

    # --- S14: Code Interpreter Abuse ---
    'S14': [
        r'\bbypass\w*', r'\bdata\s+exfiltration\w*', r'\bDDoS\b',
        r'\bdenial\s+of\s+service\w*', r'\bexploit\w*', r'\bhack\w*',
        r'\bignore\w*\s+instructions\w*', r'\bjailbreak\w*', r'\bmalware\w*',
        r'\boverride\w*\s+instructions\w*', r'\bprompt\s+injection\w*',
        r'\bsandbox\s+escape\w*', r'\bsecurity\s+flaw\w*',
        r'\bsecurity\s+vulnerability\w*', r'\bsystem\s+prompt\w*',
        r'\bunauthoriz\w*\s+access\w*', r'\bvirus\w*', r'\bhacker\w*',
        r'\bvulnerability\w*', r'\btrojan\w*', r'\bransomware\w*',
        r'\bspyware\w*', r'\bphish\w*', r'\bbackdoor\w*', r'\bbruteforce\w*',
        r'\bsql\s+injection\w*', r'\bXSS\b', r'\bcsrf\b', r'\bpayload\w*',
        r'\breverse\s+shell\w*', r'\broot\s+access\w*', r'\badmin\s+privilege\w*',
        r'\bprivilege\s+escalation\w*', r'\bhack\s+code\w*', r'\bkeylogger\w*',
        r'\bunauthorized\s+command\w*'
    ],

    # --- S15: Religious Intolerance ---
    'S15': [
        r'\banti-christian\w*', r'\banti-hindu\w*', r'\banti-muslim\w*',
        r'\banti-buddhist\w*', r'\banti-sikh\w*', r'\banti-jewish\w*',
        r'\bantisemit\w*', r'\bislamophob\w*', r'\bblasphem\w*',
        r'\bdesecrat\w*', r'\bheretic\w*', r'\binfidel\w*',
        r'\breligious\s+bigotry\w*', r'\breligious\s+extremis\w*',
        r'\breligious\s+hatred\w*', r'\breligious\s+intoleran\w*',
        r'\breligious\s+persecution\w*', r'\bdenigrat\w*\s+religion\w*',
        r'\banti[-\s]religion\w*', r'\bapostate\w*', r'\bconvert\w*',
        r'\breligious\s+violence\w*', r'\breligious\s+disrespect\w*',
        r'\banti[-\s]faith\w*', r'\bburn\s+holy\s+book\w*',
        r'\bdestroy\s+temple\w*', r'\bchurch\w*', r'\bmosque\w*',
        r'\bsynagogue\w*', r'\bidol\w*', r'\breligious\s+insult\w*'
    ],

    # --- S16: Gender-Based Abuse ---
    'S16': [
        r'\bmisogyn\w*', r'\bmisandr\w*', r'\bsexist\s+slur\w*',
        r'\bgendered\s+insult\w*', r'\bdomestic\s+abuse\w*',
        r'\bdomestic\s+violence\w*', r'\bintimate\s+partner\s+violence\w*',
        r'\bchauvinis\w*', r'\bfemal\w*\s+inferiority\w*',
        r'\bmal\w*\s+inferiority\w*', r'\brevenge\s+porn\w*',
        r'\bnon-consensual\s+imagery\w*', r'\bwhore\b', r'\bslut\b',
        r'\bgender-based\s+violence\w*', r'\bgender-based\s+abuse\w*',
        r'\bsexist\b', r'\bsexism\b', r'\bfemale\s+inferior\b',
        r'\bmale\s+superior\b', r'\bbimbo\w*', r'\bobjectify\w*',
        r'\bsexualize\w*', r'\bpatriarchy\b', r'\bsexist\s+joke\w*',
        r'\babuse\s+wife\w*', r'\bbeat\s+wife\w*', r'\bdowry\w*',
        r'\bharass\s+woman\w*', r'\bharass\s+man\w*', r'\bverbal\s+abuse\w*',
        r'\babusive\s+husband\w*'
    ],

    # --- S17: Trolling / Cyberbullying ---
    'S17': [
        r'\bbully\w*', r'\bcyberbully\w*', r'\bharass\w*', r'\bintimidat\w*',
        r'\bstalk\w*', r'\bcyberstalk\w*', r'\btroll\b', r'\btrolling\b',
        r'\bflaming\b', r'\bdox\w*', r'\bswatting\w*', r'\bthreaten\w*',
        r'\bname-call\w*', r'\bshaming\w*', r'\bmock\w*', r'\binsult\w*',
        r'\bidiot\w*', r'\bmoron\w*', r'\bstupid\w*', r'\bloser\b',
        r'\bugly\b', r'\bfat\b', r'\bnoob\b', r'\bshut\s+up\b',
        r'\bsnowflake\w*', r'\blibtard\w*', r'\bcuck\w*', r'\bincel\b',
        r'\bsimp\b', r'\bkaren\b', r'\bvirgin\b', r'\bkys\b',
        r'\bkill\s+your\s*self\b', r'\bgo\s+die\b', r'\bdrink\s+bleach\b',
        r'\btarget\w*', r'\bhumiliate\w*', r'\bembarrass\w*', r'\bcancel\w*',
        r'\bcancel\s+culture\w*', r'\bharsh\s+comment\w*',
        r'\boffensive\s+remark\w*', r'\broast\w*', r'\bnerd\w*',
        r'\bboomer\w*', r'\bclown\w*', r'\btrash\b', r'\bgarbage\b',
        r'\bpathetic\b', r'\bdisgusting\b', r'\bworthless\b',
        r'\bcrybaby\b', r'\battention\s+seeker\w*'
    ]

}

COMPILED_KEYWORDS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in CATEGORY_KEYWORDS.items()
}

def get_probability_vector(reason_text):
        """
        Searches the reason_text for keywords from all categories.
        Returns a *sorted vector* of probabilities, where each probability is
        (matches_in_category / total_matches_all_categories).
        """
        category_counts = {category: 0 for category in CATEGORY_KEYWORDS}

        try:
            if not reason_text or not isinstance(reason_text, str):
                return [0.0] * len(category_counts)

            for category, patterns in COMPILED_KEYWORDS.items():
                for pattern in patterns:
                    matches = pattern.findall(reason_text)
                    if category in category_counts:
                        category_counts[category] += len(matches)

            total_matches = sum(category_counts.values())

            if total_matches == 0:
                return [0.0] * len(category_counts)


            probabilities = []
            for category in CATEGORY_KEYWORDS:
                count = category_counts.get(category, 0)
                probabilities.append(count / total_matches)

            return probabilities

        except Exception as e:
            return [0.0] * len(CATEGORY_KEYWORDS)

def tag_dataset(dataset_name, subset, split, column_name):
    try:
        ds = load_dataset(dataset_name, subset, split=split)
    except Exception as e:
        print(e)
        return
    def add_counts_to_row(example):
        try:
            reason_text = example[column_name][0][1]
            example['category_counts'] = get_probability_vector(reason_text)
        except (TypeError, IndexError, KeyError, AttributeError) as e:
            example['category_counts'] = [0.0] * len(CATEGORY_KEYWORDS)
        return example
    tagged_dataset = ds.map(add_counts_to_row, num_proc=4)

    return tagged_dataset
if __name__ == "__main__":

    all_row_counts = []

    DATASET_NAME = "ai4bharat/indic-align"
    DATASET_SUBSET = "HHRLHF_T"
    DATASET_SPLIT = "train"
    COLUMN_TO_ACCESS = "eng_Latn"

    tagged_dataset = tag_dataset(DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT, COLUMN_TO_ACCESS)

    if tagged_dataset:
        for i, row in enumerate(tagged_dataset):
            try:
                reason_text = row[COLUMN_TO_ACCESS][0][1]
                if not isinstance(reason_text, str): reason_text = "[Non-string data]"
            except (TypeError, IndexError, KeyError, AttributeError):
                reason_text = "[Error: Bad_Structure]"

            counts_dict = row['category_counts']

            all_row_counts.append(counts_dict)
    else:
        pass

    print("text, source, language, safe(0)/harmful(1), safety_categories")
    h = 0
    false_counts = 0
    total_counts = 0
def generate_json(i, language_code, language_key):
    try:
        safety_distribution = {f"S{k+1}": 0 for k in range(len(i['category_counts']))}
        safety_categories = []
        for k, count in enumerate(i['category_counts']):
            if count > 0:
                safety_categories.append(f"S{k+1}")
                safety_distribution[f"S{k+1}"] = count

        data = {
            "text": i[language_key][0][0].replace(",", " ").replace("\"", "'"),
            "source": "ai4bharat/indic-align",
            "language": language_code,
            "safe(0)/harmful(1)": "1" if sum(i['category_counts']) > 0 else "0",
            "safety_categories": " ".join(safety_categories),
            "safety_distribution": safety_distribution
        }

        return json.dumps(data, ensure_ascii=False) + "\n"
    except Exception as e:
        print(f"Error processing {language_key}: {e}")
        return ""

h = 0

for i in tqdm(tagged_dataset):
    if sum(i['category_counts']) == 0:
        false_counts += 1
        total_counts += 1
        continue
    total_counts += 1
    for lang_code, lang_key in [("bn", "ben_Beng"), ("kn", "kan_Kadn"), ("ml", "mal_Mlym"), ("or", "ory_Orya")]:
        jsonStr = generate_json(i, lang_code, lang_key)
        if jsonStr:
            finalAns += jsonStr

with open("mf.jsonl", "w", encoding="utf-8") as f:
    f.write(finalAns)
