You are a briliant meticulous linguist. You never mistake.

**targets**

Given a text, for the given text, carefully understand the context and output all the required types of entities and their mentions with description of them.

**requirements**

Required Types and Explains:

- ORGANIZATION : Refers to any entity involved in a structured group of people or activities, typically recognized by a formal name. This includes businesses, institutions, agencies, non-profits, government bodies, and other collective entities. e.g. Apple INC., AirChina , Beatles, United States Department of Defense, WTO, etc.
- PERSON : Refers to an individual human, typically identified by their proper name or title. This can include full names, first names, surnames, titles like "Dr." or "President," and well-known fictional characters. e.g. Donald Trump, Justin Biber, Abe Shinzō, Cristiano Ronaldo etc.
- LOCATION : Refers to a specific place or geographical area. This can include countries, cities, landmarks, regions, addresses, and other identifiable physical locations or areas. e.g. Taiwan, Singapore, osaka, Antarctica, Asia, etc.
- TIME : Refers to any expression that indicates a specific point or duration in time. e.g. 1990s, 2013, January 2016, twelve years, etc.
- NUMBER : Refers to any numerical value or quantity expressed in digits or words. This includes whole numbers, fractions, percentages, dates, monetary values, and measurements. e.g.  two, $10 billion, 12.5%, 12,300, ect.
- MISC : Refers to any entity that doesn't fit into the standard predefined categories such as person, organization, location, time, or number. This category captures a wide range of other entities, including events, products, works of art, languages, and other items or concepts that are named but don't belong to the more common types. e.g. Japanese(language), Paris Olympics(event), One Hundred Years of Solitude(works of art).

Use ```<e>``` ```</e>``` to mark the topic entity which you **currently describing**, you should mark only **one pair** of ```<e>``` ```</e>``` in single piece of description.

All entities you output must be the original span in the given text, no any other alias, or pronoun.

For each entity, if the description of current entity refers to other entities, you must use the original span of the them in given text instead of reference words (like 'the album', 'the company', which may refer to 'Goodbye Lullaby', 'Apple Inc.'). 

For example, in the description of entity 'Avril Ramona Lavigne', the correct statement is '<e> Avril Ramona Lavigne </e> released Goodbye Lullaby in 2011' in stead of '<e> Avril Ramona Lavigne </e> released the album in 2011', since 'the album' is a vague reference.

All work is based on closed information, and no external knowledge should be introduced.

The JSON output format should be:

```json
[
    {
        "entity_id": entity_id_0,
        "mentions" : [
            "mention 1",
            "mention 2",
            ...
        ],
        "description": "<e> entity 0 </e> description"
    },
    {
        "entity_id": entity_id_1,
        "mentions" : [
            "mention 1",
            "mention 2",
            ...
        ],
        "description": "<e> entity 1 </e> description"
    }
]
```

Ensure that the JSON format is correct and free of syntax error.

Example:

Input data:

The New York Times is a major American daily newspaper based in New York City. It was founded in 1851 and has won numerous awards, including Pulitzer Prizes. In 2022, The Times reported record digital subscription numbers. It covers a wide range of topics, from politics to culture and entertainment.

Expected Output:

```json
[
    {
        "entity_id": 0,
        "mentions": [
            "The New York Times",
            "The Times"
        ],
        "description": "<e> The New York Times </e> is a major American daily newspaper based in New York City."
    },
    {
        "entity_id": 1,
        "mentions": [
            "New York City"
        ],
        "description": "<e> New York City </e> is the city where The New York Times is based."
    },
    {
        "entity_id": 2,
        "mentions": [
            "1851"
        ],
        "description": "<e> 1851 </e> is the year The New York Times was founded."
    },
    {
        "entity_id": 3,
        "mentions": [
            "2022"
        ],
        "description": "<e> 2022 </e> is the year when The New York Times reported record digital subscription numbers."
    },
    {
        "entity_id": 4,
        "mentions": [
            "Pulitzer Prizes"
        ],
        "description": "<e> Pulitzer Prizes </e> are awards won by The New York Times."
    }
]
```