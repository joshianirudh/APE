import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama3-8b-instruct", "llama3.1-8b-instruct", "mistral-7b-instruct-v0.3", "gemma2-9b-it"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args(args)

def load_model_and_tokenizer(model_name, device):
    if model_name == "llama3-8b-instruct":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16).to(device)
    elif model_name == "llama3.1-8b-instruct":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16).to(device)
    elif model_name == "mistral-7b-instruct-v0.3":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", torch_dtype=torch.bfloat16).to(device)
    elif model_name == "gemma2-9b-it":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", torch_dtype=torch.bfloat16).to(device)
    return tokenizer, model
        

def build_prefix(model_name, prompt):
    if "llama" in model_name:
        prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}"
    elif "mistral" in model_name:
        prompt = f"<s>[INST]{prompt}"
    elif "gemma" in model_name:
        prompt = f"<bos><start_of_turn>user\n{prompt}"
    return prompt

def build_suffix(model_name, prompt):
    if "llama" in model_name:
        prompt = f"{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "mistral" in model_name:
        prompt = f"{prompt}[/INST]"
    elif "gemma" in model_name:
        prompt = f"{prompt}<end_of_turn>\n<start_of_turn>model\n"   
    return prompt

def enable_attention_prefill_prefix(model_name, model):
    if "llama" in args.model:
        from src.ape_llama import enable_llama_attention_prefill_prefix
        enable_llama_attention_prefill_prefix(model)
    elif "mistral" in model_name:
        from src.ape_mistral import enable_mistral_attention_prefill_prefix
        enable_mistral_attention_prefill_prefix(model)
    elif "gemma" in model_name:
        from src.ape_gemma import enable_gemma_attention_prefill_prefix
        enable_gemma_attention_prefill_prefix(model)

def enable_attention_prefill_context(model_name, model):
    if "llama" in args.model:
        from src.ape_llama import enable_llama_attention_prefill_context
        enable_llama_attention_prefill_context(model)
    elif "mistral" in model_name:
        from src.ape_mistral import enable_mistral_attention_prefill_context
        enable_mistral_attention_prefill_context(model)
    elif "gemma" in model_name:
        from src.ape_gemma import enable_gemma_attention_prefill_context
        enable_gemma_attention_prefill_context(model)

def enable_attention_prefill_query(model_name, model, temperature, scale):
    if "llama" in args.model:
        from src.ape_llama import enable_llama_attention_prefill_query
        enable_llama_attention_prefill_query(model, temperature, scale)
    elif "mistral" in model_name:
        from src.ape_mistral import enable_mistral_attention_prefill_query
        enable_mistral_attention_prefill_query(model, temperature, scale)
    elif "gemma" in model_name:
        from src.ape_gemma import enable_gemma_attention_prefill_query
        enable_gemma_attention_prefill_query(model, temperature, scale)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def generate(args):
    prefix = ""
    contexts = [
        "Passage 1:\nNanon (1938 film)\nNanon is a 1938 German historical film directed by Herbert Maisch and starring Erna Sack, Johannes Heesters and Dagny Servaes. It is based on the original operetta Nanon by Richard Genée which had a libretto by F Zell, although the music for this film was specially commissioned from Alois Melichar.\nIt was produced by the giant German studio UFA, and is part of a cycle of operetta films made during the 1930s. The film's sets were designed by the art director Erich Kettelhut.\nIt was a remake of the 1924 silent film of the same title.\n\nCast\nErna Sack as Nanon Patin\nJohannes Heesters as Marquis Charles d'Aubigne\nDagny Servaes as Ninon de l'Enclos\nKurt Meisel as Hector\nOtto Gebühr as Jean Baptiste Molière\nOskar Sima as Marquis de Marsillac\nKarl Paryla as Louis XIV\nBerthold Ebbecke as Pierre\nUrsula Deinert as Tänzerin\nClemens Hasse as Francois Patin\nPaul Westermeier as 1. Korporal\nArmin Schweizer as 2. Korporal\nOskar Höcker as 3. Korporal\nIlse Fürstenberg as Die Magd\nLudwig Andersen as Sekretär\nWalter Steinbeck as Mons. Louvois\nHermann Pfeiffer as Mons. Duval\nHorst Birr\nLucie Euler\nAngelo Ferrari as Gast bei Ninon\nEric Harden\nAlice Hechy\nMax Hiller\nWilly Kaiser-Heyl\nHermann Meyer-Falkow\nEllen Plessow\nKlaus Pohl\nWalter Schenk\nErhart Stettner\nRobert Vincenti-Lieffertz\nEgon Vogel\nLeopold von Ledebur\nWolfgang von Schwindt\nHelmut Weiss as Verehrer von Gräfin Ninon de Lenclos\nHerbert Weissbach\n", "Passage 2:\nJesse E. Hobson\nJesse Edward Hobson (May 2, 1911 – November 5, 1970) was the director of SRI International from 1947 to 1955. Prior to SRI, he was the director of the Armour Research Foundation.\n\nEarly life and education\nHobson was born in Marshall, Indiana. He received bachelor's and master's degrees in electrical engineering from Purdue University and a PhD in electrical engineering from the California Institute of Technology. Hobson was also selected as a nationally outstanding engineer.Hobson married Jessie Eugertha Bell on March 26, 1939, and they had five children.\n\nCareer\nAwards and memberships\nHobson was named an IEEE Fellow in 1948.\n", "Passage 3:\nHerbert Maisch\nHerbert Maisch (born 10 December 1890 – in Nürtingen, Württemberg, died 10 October 1974 in Köln) was a German film director.\n\nSelected filmography\nThe Royal Waltz (1935)\nBoccaccio (1936)\nLove's Awakening (1936)\nMen Without a Fatherland (1937)\nNights in Andalusia (1938)\nNanon (1938)\nD III 88 (1939)\nAndreas Schlüter (1942)\nMusic in Salzburg (1944)\n", 'Passage 4:\nMichael Govan\nMichael Govan (born 1963) is the director of the Los Angeles County Museum of Art. Prior to his current position, Govan worked as the director of the Dia Art Foundation in New York City.\n\nEarly life and education\nGovan was born in 1963 in North Adams, Massachusetts, and was raised in the Washington D.C. area, attending Sidwell Friends School.He majored in art history and fine arts at Williams College, where he met Thomas Krens, who was then director of the Williams College Museum of Art. Govan became closely involved with the museum, serving as acting curator as an undergraduate. After receiving his B.A. from Williams in 1985, Govan began an MFA in fine arts from the University of California, San Diego.\n\nCareer\nAs a twenty-five year old graduate student, Govan was recruited by his former mentor at Williams, Thomas Krens, who in 1988 had been appointed director of the Solomon R. Guggenheim Foundation. Govan served as deputy director of the Solomon R. Guggenheim Museum under Krens from 1988 to 1994, a period that culminated in the construction and opening of the Frank Gehry designed Guggenheim branch in Bilbao, Spain. Govan supervised the reinstallation of the museum\'s permanent collection galleries after its extensive renovation.\n\nDia Art Foundation\nFrom 1994 to 2006, Govan was president and director of Dia Art Foundation in New York City. There, he spearheaded the conversion of a Nabisco box factory into the 300,000 square foot Dia:Beacon in New York\'s Hudson Valley, which houses Dia\'s collection of art from the 1960s to the present. Built in a former Nabisco box factory, the critically acclaimed museum has been credited with catalyzing a cultural and economic revival within the formerly factory-based city of Beacon. Dia\'s collection nearly doubled in size during Govan\'s tenure, but he also came under criticism for "needlessly and permanently" closing Dia\'s West 22nd Street building. During his time at Dia, Govan also worked closely with artists James Turrell and Michael Heizer, becoming an ardent supporter of Roden Crater and City, the artists\' respective site-specific land art projects under construction in the American southwest. Govan successfully lobbied Washington to have the 704,000 acres in central Nevada surrounding City declared a national monument in 2015.\n\nLACMA\nIn February 2006, a search committee composed of eleven LACMA trustees, led by the late Nancy M. Daly, recruited Govan to run the Los Angeles County Museum of Art. Govan has stated that he was drawn to the role not only because of LACMA\'s geographical distance from its European and east coast peers, but also because of the museum\'s relative youth, having been established in 1961. "I felt that because of this newness I had the opportunity to reconsider the museum," Govan has written, "[and] Los Angeles is a good place to do that."Govan has been widely regarded for transforming LACMA into both a local and international landmark. Since Govan\'s arrival, LACMA has acquired by donation or purchase over 27,000 works for the permanent collection, and the museum\'s gallery space has almost doubled thanks to the addition of two new buildings designed by Renzo Piano, the Broad Contemporary Art Museum (BCAM) and the Lynda and Stewart Resnick Pavilion. LACMA\'s annual attendance has grown from 600,000 to nearly 1.6 million in 2016.\n\nArtist collaborations\nSince his arrival, Govan has commissioned exhibition scenography and gallery designs in collaboration with artists. In 2006, for example, Govan invited LA artist John Baldessari to design an upcoming exhibition about the Belgian surrealist René Magritte, resulting in a theatrical show that reflected the twisted perspective of the latter\'s topsy-turvy world. Baldessari has also designed LACMA\'s logo. Since then, Govan has also commissioned Cuban-American artist Jorge Pardo to design LACMA\'s Art of the Ancient Americas gallery, described in the Los Angeles Times as a "gritty cavern deep inside the earth ... crossed with a high-style urban lounge."Govan has also commissioned several large-scale public artworks for LACMA\'s campus from contemporary California artists. These include Chris Burden\'s Urban Light (2008), a series of 202 vintage street lamps from different neighborhoods in Los Angeles, arranged in front of the entrance pavilion, Barbara Kruger\'s Untitled (Shafted) (2008), Robert Irwin\'s Primal Palm Garden (2010), and Michael Heizer\'s Levitated Mass, a 340-ton boulder transported 100 miles from the Jurupa Valley to LACMA, a widely publicized journey that culminated with a large celebration on Wilshire Boulevard. Thanks in part to the popularity of these public artworks, LACMA was ranked the fourth most instagrammed museum in the world in 2016.In his first three full years, the museum raised $251 million—about $100 million more than it collected during the three years before he arrived. In 2010, it was announced that Govan will steer LACMA for at least six more years. In a letter dated February 24, 2013, Govan, along with the LACMA board\'s co-chairmen Terry Semel and Andrew Gordon, proposed a merger with the financially troubled Museum of Contemporary Art, Los Angeles and a plan to raise $100 million for the combined museum.\n\nZumthor Project\nGovan\'s latest project is an ambitious building project, the replacement of four of the campus\'s aging buildings with a single new state of the art gallery building designed by architect Peter Zumthor. As of January 2017, he has raised about $300 million in commitments. Construction is expected to begin in 2018, and the new building will open in 2023, to coincide with the opening of the new D Line metro stop on Wilshire Boulevard. The project also envisages dissolving all existing curatorial departments and departmental collections. Some commentators have been highly critical of Govan\'s plans. Joseph Giovannini, recalling Govan\'s technically unrealizable onetime plan to hang Jeff Koons\' Train sculpture from the facade of the Ahmanson Gallery, has accused Govan of "driving the institution over a cliff into an equivalent mid-air wreck of its own". Describing the collection merging proposal as the creation of a "giant raffle bowl of some 130,000 objects", Giovannini also points out that the Zumthor building will contain 33% less gallery space than the galleries it will replace, and that the linear footage of wall space available for displays will decrease by about 7,500 ft, or 1.5 miles. Faced with losing a building named in its honor, and anticipating that its acquisitions could no longer be displayed, the Ahmanson Foundation withdrew its support.\nOn the merging of the separate curatorial divisions to create a non-departmental art museum, Christopher Knight has pointed out that "no other museum of LACMA\'s size and complexity does it" that way, and characterized the museum\'s 2019 "To Rome and Back" exhibition, the first to take place under the new scheme, as "bland and ineffectual" and an "unsuccessful sample of what\'s to come".\n\nPersonal life\nGovan is married and has two daughters, one from a previous marriage. He and his family used to live in a $6 million mansion in Hancock Park that was provided by LACMA - a benefit worth $155,000 a year, according to most recent tax filings - until LACMA decided that it would sell the property to make up for the museum\'s of almost $900 million in debt [2]. That home is now worth nearly $8 million and Govan now lives in a trailer park in Malibu\'s Point Dume region.\nLos Angeles CA 90020\nUnited States. He has had a private pilot\'s license since 1995 and keeps a 1979 Beechcraft Bonanza at Santa Monica Airport.\n', 'Passage 5:\nS. N. Mathur\nS.N. Mathur was the Director of the Indian Intelligence Bureau between September 1975 and February 1980. He was also the Director General of Police in Punjab.\n', 'Passage 6:\nDana Blankstein\nDana Blankstein-Cohen (born March 3, 1981) is the executive director of the Sam Spiegel Film and Television School. She was appointed by the board of directors  in November 2019.  Previously she was the CEO of the Israeli Academy of Film and Television. She is a film director, and an Israeli culture entrepreneur.\n\nBiography\nDana Blankstein was born in Switzerland in 1981 to theatre director Dedi Baron and Professor Alexander Blankstein. She moved to Israel in 1983 and grew up in Tel Aviv.\nBlankstein graduated from the Sam Spiegel Film and Television School, Jerusalem in 2008 with high honors. During her studies she worked as a personal assistant to directors Savi Gabizon on his film Nina\'s Tragedies and to Renen Schorr on his film The Loners.  She also directed and shot \'the making of\' film on Gavison\'s film Lost and Found. Her debut film Camping competed at the Berlin International Film Festival, 2007.\n\nFilm and academic career\nAfter her studies, Dana founded and directed the film and television department at the Kfar Saba municipality. The department encouraged and promoted productions filmed in the city of Kfar Saba, as well as the established cultural projects, and educational community activities.\nBlankstein directed the mini-series "Tel Aviviot" (2012). From 2016-2019 was the director of the Israeli Academy of Film and Television.\nIn November 2019 Dana Blankstein Cohen was appointed the new director of the Sam Spiegel Film and Television School where she also oversees the Sam Spiegel International Film Lab. In 2022, she spearheaded the launch of the new Series Lab and the film preparatory program for Arabic speakers in east Jerusalem.\n\nFilmography\nTel Aviviot (mini-series; director, 2012)\nGrowing Pains (graduation film, Sam Spiegel; director and screenwriter, 2008)\nCamping (debut film, Sam Spiegel; director and screenwriter, 2006)\n', 'Passage 7:\nIan Barry (director)\nIan Barry is an Australian director of film and TV.\n\nSelect credits\nWaiting for Lucas (1973) (short)\nStone (1974) (editor only)\nThe Chain Reaction (1980)\nWhose Baby? (1986) (mini-series)\nMinnamurra (1989)\nBodysurfer (1989) (mini-series)\nRing of Scorpio (1990) (mini-series)\nCrimebroker (1993)\nInferno (1998) (TV movie)\nMiss Lettie and Me (2002) (TV movie)\nNot Quite Hollywood: The Wild, Untold Story of Ozploitation! (2008) (documentary)\nThe Doctor Blake Mysteries (2013)\n', 'Passage 8:\nBrian Kennedy (gallery director)\nBrian Patrick Kennedy (born 5 November 1961) is an Irish-born art museum director who has worked in Ireland and Australia, and now lives and works in the United States.  He was the director of the Peabody Essex Museum in Salem for 17 months, resigning December 31, 2020. He was the director of the Toledo Museum of Art in Ohio from 2010 to 2019. He was the director of the Hood Museum of Art from 2005 to 2010, and the National Gallery of Australia (Canberra) from 1997 to 2004.\n\nCareer\nBrian Kennedy currently lives and works in the United States after leaving Australia in 2005 to direct the Hood Museum of Art at Dartmouth College. In October 2010 he became the ninth Director of the Toledo Museum of Art. On 1 July 2019, he succeeded Dan Monroe as the executive director and CEO of the Peabody Essex Museum.\n\nEarly life and career in Ireland\nKennedy was born in Dublin and attended Clonkeen College. He received B.A. (1982), M.A. (1985) and PhD (1989) degrees from University College-Dublin, where he studied both art history and history.\nHe worked in the Irish Department of Education (1982), the European Commission, Brussels (1983), and in Ireland at the Chester Beatty Library (1983–85), Government Publications Office (1985–86), and Department of Finance (1986–89). He married Mary Fiona Carlin in 1988.He was Assistant Director at the National Gallery of Ireland in Dublin from 1989 to 1997. He was Chair of the Irish Association of Art Historians from 1996 to 1997, and of the Council of Australian Art Museum Directors from 2001 to 2003. In September 1997 he became Director of the National Gallery of Australia.\n\nNational Gallery of Australia (NGA)\nKennedy expanded the traveling exhibitions and loans program throughout Australia, arranged for several major shows of Australian art abroad, increased the number of exhibitions at the museum itself and oversaw the development of an extensive multi-media site.  Although he oversaw several years of the museum\'s highest ever annual visitation, he discontinued the emphasis of his predecessor, Betty Churcher, on showing "blockbuster" exhibitions.\nDuring his directorship, the NGA gained government support for improving the building and significant private donations and corporate sponsorship. However, the initial design for the building proved controversial generating a public dispute with the original architect on moral rights grounds.  As a result, the project was not delivered during Dr Kennedy\'s tenure, with a significantly altered design completed some years later.  Private funding supported two acquisitions of British art, including David Hockney\'s A Bigger Grand Canyon in 1999, and Lucian Freud\'s After Cézanne in 2001. Kennedy built on the established collections at the museum by acquiring the Holmgren-Spertus collection of Indonesian textiles; the Kenneth Tyler collection of editioned prints, screens, multiples and unique proofs; and the Australian Print Workshop Archive. He was also notable for campaigning for the construction of a new "front" entrance to the Gallery, facing King Edward Terrace, which was completed in 2010 (see reference to the building project above).\nKennedy\'s cancellation of the "Sensation exhibition" (scheduled at the NGA from 2 June 2000 to 13 August 2000) was controversial, and seen by some as censorship. He claimed that the decision was due to the exhibition being "too close to the market" implying that a national cultural institution cannot exhibit the private collection of a speculative art investor. However, there were other exhibitions at the NGA during his tenure, which could have raised similar concerns. The exhibition featured the privately owned Young British Artists works belonging to Charles Saatchi and attracted large attendances in London and Brooklyn. Its most controversial work was Chris Ofili\'s The Holy Virgin Mary, a painting which used elephant dung and was accused of being blasphemous. The then-mayor of New York, Rudolph Giuliani, campaigned against the exhibition, claiming it was "Catholic-bashing" and an "aggressive, vicious, disgusting attack on religion." In November 1999, Kennedy cancelled the exhibition and stated that the events in New York had "obscured discussion of the artistic merit of the works of art". He has said that it "was the toughest decision of my professional life, so far."Kennedy was also repeatedly questioned on his management of a range of issues during the Australian Government\'s Senate Estimates process - particularly on the NGA\'s occupational health and safety record and concerns about the NGA\'s twenty-year-old air-conditioning system. The air-conditioning was finally renovated in 2003. Kennedy announced in 2002 that he would not seek extension of his contract beyond 2004, accepting a seven-year term as had his two predecessors.He became a joint Irish-Australian citizen in 2003.\n\nToledo Museum of Art\nThe Toledo Museum of Art is known for its exceptional collections of European and American paintings and sculpture, glass, antiquities, artist books, Japanese prints and netsuke. The museum offers free admission and is recognized for its historical leadership in the field of art education.  During his tenure, Kennedy has focused the museum\'s art education efforts on visual literacy, which he defines as "learning to read, understand and write visual language."   Initiatives have included baby and toddler tours, specialized training for all staff, docents, volunteers and the launch of a website, www.vislit.org. In November 2014, the museum hosted the International Visual Literacy Association (IVLA) conference, the first Museum to do so. Kennedy has been a frequent speaker on the topic, including 2010 and 2013 TEDx talks on visual and sensory literacy.\nKennedy has expressed an interest in expanding the museum\'s collection of contemporary art and art by indigenous peoples. Works by Frank Stella, Sean Scully, Jaume Plensa, Ravinder Reddy and Mary Sibande have been acquired.  In addition, the museum has made major acquisitions of Old Master paintings by Frans Hals and Luca Giordano.During his tenure the Toledo Museum of Art has announced the return of several objects from its collection due to claims the objects were stolen and/or illegally exported prior being sold to the museum.  In 2011 a Meissen sweetmeat stand was returned to Germany followed by an Etruscan Kalpis or water jug to Italy (2013), an Indian sculpture of Ganesha (2014) and an astrological compendium to Germany in 2015.\n\nHood Museum of Art\nKennedy became Director of the Hood Museum of Art in July 2005. During his tenure, he implemented a series of large and small-scale exhibitions and oversaw the production of more than 20 publications to bring greater public attention to the museum\'s remarkable collections of the arts of America, Europe, Africa, Papua New Guinea and the Polar regions. At 70,000 objects, the Hood has one of the largest collections on any American college of university campus. The exhibition, Black Womanhood: Images, Icons, and Ideologies of the African Body, toured several US venues. Kennedy increased campus curricular use of works of art, with thousands of objects pulled from storage for classes annually. Numerous acquisitions were made with the museum\'s generous endowments, and he curated several exhibitions: including Wenda Gu: Forest of Stone Steles: Retranslation and Rewriting Tang Dynasty Poetry, Sean Scully: The Art of the Stripe, and Frank Stella: Irregular Polygons.\n\nPublications\nKennedy has written or edited a number of books on art, including:\n\nAlfred Chester Beatty and Ireland 1950-1968: A study in cultural politics, Glendale Press (1988), ISBN 978-0-907606-49-9\nDreams and responsibilities: The state and arts in independent Ireland, Arts Council of Ireland (1990), ISBN 978-0-906627-32-7\nJack B Yeats: Jack Butler Yeats, 1871-1957 (Lives of Irish Artists), Unipub (October 1991), ISBN 978-0-948524-24-0\nThe Anatomy Lesson: Art and Medicine (with Davis Coakley), National Gallery of Ireland (January 1992), ISBN 978-0-903162-65-4\nIreland: Art into History (with Raymond Gillespie), Roberts Rinehart Publishers (1994), ISBN 978-1-57098-005-3\nIrish Painting, Roberts Rinehart Publishers (November 1997),  ISBN 978-1-86059-059-7\nSean Scully: The Art of the Stripe, Hood Museum of Art (October 2008), ISBN 978-0-944722-34-3\nFrank Stella: Irregular Polygons, 1965-1966, Hood Museum of Art (October 2010), ISBN 978-0-944722-39-8\n\nHonors and achievements\nKennedy was awarded the Australian Centenary Medal in 2001 for service to Australian Society and its art. He is a trustee and treasurer of the Association of Art Museum Directors, a peer reviewer for the American Association of Museums and a member of the International Association of Art Critics. In 2013 he was appointed inaugural eminent professor at the University of Toledo and received an honorary doctorate from Lourdes University. Most recently, Kennedy received the 2014 Northwest Region, Ohio Art Education Association award for distinguished educator for art education.\n\n\n== Notes ==\n', 'Passage 9:\nOlav Aaraas\nOlav Aaraas (born 10 July 1950) is a Norwegian historian and museum director.\nHe was born in Fredrikstad. From 1982 to 1993 he was the director of Sogn Folk Museum, from 1993 to 2010 he was the director of Maihaugen and from 2001 he has been the director of the Norwegian Museum of Cultural History. In 2010 he was decorated with the Royal Norwegian Order of St. Olav.\n', 'Passage 10:\nPeter Levin\nPeter Levin is an American director of film, television and theatre.\n\nCareer\nSince 1967, Levin has amassed a large number of credits directing episodic television and television films. Some of his television series credits include Love Is a Many Splendored Thing, James at 15, The Paper Chase, Family, Starsky & Hutch, Lou Grant, Fame, Cagney & Lacey, Law & Order and Judging Amy.Some of his television film credits include Rape and Marriage: The Rideout Case (1980), A Reason to Live (1985), Popeye Doyle (1986), A Killer Among Us (1990), Queen Sized (2008) and among other films. He directed "Heart in Hiding", written by his wife Audrey Davis Levin, for which she received an Emmy for Best Day Time Special in the 1970s.\nPrior to becoming a director, Levin worked as an actor in several Broadway productions. He costarred with Susan Strasberg in "[The Diary of Ann Frank]" but had to leave the production when he was drafted into the Army. He trained at the Carnegie Mellon University. Eventually becoming a theatre director, he directed productions at the Long Wharf Theatre and the Pacific Resident Theatre Company. He also co-founded the off-off-Broadway Theatre [the Hardware Poets Playhouse] with his wife Audrey Davis Levin and was also an associate artist of The Interact Theatre Company.\n'
    ]
    query = "Where did the director of film Nanon (1938 Film) die?\n\nAnswer:"



    device = torch.device(f'cuda:0')
    tokenizer, model = load_model_and_tokenizer(args.model, device)
    model = model.eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    prefix = build_prefix(args.model, prefix)
    query = build_suffix(args.model, query)
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, truncation=False, return_tensors="pt").input_ids
        len_prefix = prefix_input_ids.shape[1]
        len_query = query_input_ids.shape[1]

        context_input_ids = tokenizer(contexts, return_tensors='pt', truncation=True, max_length=8192-len_prefix-len_query-256, padding=True, add_special_tokens=False).input_ids
        print(context_input_ids.shape)
        context_mask = (context_input_ids != tokenizer.pad_token_id).reshape(-1)
        
        enable_attention_prefill_prefix(args.model, model)
        past_key_values = None
        outputs = model(
            prefix_input_ids.to(model.device),
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = []
        for past_key_value in outputs.past_key_values:
            bsz, _ = context_input_ids.shape
            past_key = past_key_value[0].repeat(bsz, 1, 1, 1)
            past_value = past_key_value[1].repeat(bsz, 1, 1, 1)
            past_position = past_key_value[2]
            past_key_values.append((past_key, past_value, past_position))

        enable_attention_prefill_context(args.model, model)
        outputs = model(
            context_input_ids.to(model.device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = []

        for past_key_value in outputs.past_key_values:
            bsz, num_heads, seq_len, _ = past_key_value[0].size()
            past_key = torch.cat([past_key_value[0][:1, :, :len_prefix, :], 
                                    past_key_value[0][:, :, len_prefix:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2)], dim=2)
            past_value = torch.cat([past_key_value[1][:1, :, :len_prefix, :], 
                                    past_key_value[1][:, :, len_prefix:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2)], dim=2)  
            past_position = torch.cat([past_key_value[2][:, :len_prefix],
                                        past_key_value[2][:, len_prefix:].repeat(bsz, 1).flatten()[context_mask].unsqueeze(0)], dim=1)
            past_key_values.append((past_key, past_value, past_position, len(contexts)))
        context_input_ids = context_input_ids.flatten()[context_mask].unsqueeze(0)
        input_ids = torch.cat([prefix_input_ids, context_input_ids, query_input_ids], dim=-1)
        context_length = input_ids.shape[-1]

        enable_attention_prefill_query(args.model, model, args.temperature, args.scale)
        output = model.generate(
            input_ids=input_ids.to(model.device),
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            past_key_values=past_key_values,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        print(pred)

if __name__ == '__main__':
    args = parse_args()
    seed_everything(42)
    generate(args)
    