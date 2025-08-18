You are a briliant meticulous linguist. You never mistake.
Given an input data in the following JSON format:

```json
{
    "text": "Your text paragraph here.",
    "entities": [
        {
            "entity_id": 0,
            "mentions": [
                "mention1",
                "mention2",
                ...
            ]
        },
        ...
    ]
}
```

**Requirements**:
Please accurately and concisely output the entity description for each entity, the description should involve :
- basic information
- events
- attributes
- interaction with other entities etc.
and should be as precise and brief as possible. 

Use ```<e>``` ```</e>``` to mark the topic entity which you **currently describing**, you should mark only **one pair** of ```<e>``` ```</e>``` in single piece of description.

All entities you output must be the original span in the given text, no any other alias, or pronoun.

For each entity, if the description of current entity refers to other entities, you must use the original span of the them in given text instead of reference words (like 'the album', 'the company', which may refer to 'Goodbye Lullaby', 'Apple Inc.'). 

For example, in the description of entity 'Avril Ramona Lavigne', the correct statement is '<e> Avril Ramona Lavigne </e> released Goodbye Lullaby in 2011' in stead of '<e> Avril Ramona Lavigne </e> released the album in 2011', since 'the album' is a vague reference.

All work is based on closed information, and no external knowledge should be introduced.

The JSON output format should be:

```json
{
    "entity_description": [
        {
            "entity_id": entity_id,
            "description": "entity description"
        }
    ]
}
```

Ensure that the JSON format is correct and free of syntax errors, do not output anything beyond required.

Example:

Input data:

```json
{
    "text" : "Zest Airways , Inc. operated as AirAsia Zest ( formerly Asian Spirit and Zest Air ) , was a low - cost airline based at the Ninoy Aquino International Airport in Pasay City , Metro Manila in the Philippines . It operated scheduled domestic and international tourist services , mainly feeder services linking Manila and Cebu with 24 domestic destinations in support of the trunk route operations of other airlines . In 2013 , the airline became an affiliate of Philippines AirAsia operating their brand separately . Its main base was Ninoy Aquino International Airport , Manila . The airline was founded as Asian Spirit , the first airline in the Philippines to be run as a cooperative . On August 16 , 2013 , the Civil Aviation Authority of the Philippines ( CAAP ) , the regulating body of the Government of the Republic of the Philippines for civil aviation , suspended Zest Air flights until further notice because of safety issues . Less than a year after AirAsia and Zest Air 's strategic alliance , the airline has been rebranded as AirAsia Zest . The airline was merged into AirAsia Philippines in January 2016 . ",
    "entities": [
        {
            "entity_id": 0,
            "mentions": [
                "Zest Airways, Inc.",
                "Asian Spirit and Zest Air",
                "AirAsia Zest"
            ],
        },
        {
            "entity_id": 1,
            "mentions": [
                "Ninoy Aquino International Airport"
            ],
        },
        {
            "entity_id": 2,
            "mentions": [
                "Pasay City"
            ],
        },
        {
            "entity_id": 3,
            "mentions": [
                "Metro Manila"
            ],
        },
        {
            "entity_id": 4,
            "mentions": [
                "Philippines",
                "Republic of the Philippines"
            ],
        },
        {
            "entity_id": 5,
            "mentions": [
                "Manila"
            ],
        },
        {
            "entity_id": 6,
            "mentions": [
                "Cebu"
            ],
        },
        {
            "entity_id": 7,
            "mentions": [
                "24"
            ],
        },
        {
            "entity_id": 8,
            "mentions": [
                "2013",
                "August 16, 2013"
            ]
        },
        {
            "entity_id": 9,
            "mentions": [
                "Philippines AirAsia"
            ],
        },
        {
            "entity_id": 10,
            "mentions": [
                "Asian Spirit"
            ],
        },
        {
            "entity_id": 11,
            "mentions": [
                "Civil Aviation Authority of the Philippines",
                "CAAP"
            ],
        },
        {
            "entity_id": 12,
            "mentions": [
                "Zest Air"
            ],
        },
        {
            "entity_id": 13,
            "mentions": [
                "a year"
            ],
        },
        {
            "entity_id": 14,
            "mentions": [
                "AirAsia"
            ],
        },
        {
            "entity_id": 15,
            "mentions": [
                "AirAsia Philippines"
            ],
        },
        {
            "entity_id": 16,
            "mentions": [
                "January 2016"
            ],
        }
    ]
}
```
Expected output:

```json
[
    {
        "entity_id": 0,
        "description": "<e> Zest Airways, Inc. </e>, also known as Asian Spirit and Zest Air, was a low-cost airline based at Ninoy Aquino International Airport in Pasay City , Metro Manila , Philippines ."
    },
    {
        "entity_id": 1,
        "description": "<e> Ninoy Aquino International Airport </e> is an airport located in Pasay City , Metro Manila ,  Philippines ."
    },
    {
        "entity_id": 2,
        "description": "<e> Pasay City </e> is a city in Metro Manila , Philippines ."
    },
    {
        "entity_id": 3,
        "description": "<e> Metro Manila </e> is a metropolitan region in Philippines ."
    },
    {
        "entity_id": 4,
        "description": "<e> Philippines </e> is the country where Zest Airways, Inc. was based."
    },
    {
        "entity_id": 5,
        "description": "<e> Manila </e> is a city where Zest Airways, Inc. operated services."
    },
    {
        "entity_id": 6,
        "description": "<e> Cebu </e> is a city where Zest Airways, Inc. operated services."
    },
    {
        "entity_id": 7,
        "description": "<e> 24 </e> is the number of domestic destinations served by Zest Airways, Inc. ."
    },
    {
        "entity_id": 8,
        "description": "<e> 2013 </e> is the year when Zest Airways, Inc. became an affiliate of Philippines AirAsia and had its flights suspended by Civil Aviation Authority of the Philippines ."
    },
    {
        "entity_id": 9,
        "description": "<e> Philippines AirAsia </e> is an airline that Zest Airways, Inc. became an affiliate of in 2013 ."
    },
    {
        "entity_id": 10,
        "description": "<e> Asian Spirit </e> is the original name under which Zest Airways, Inc. was founded."
    },
    {
        "entity_id": 11,
        "description": "<e> Civil Aviation Authority of the Philippines </e> is the regulatory body that suspended Zest Air 's flights in 2013 ."
    },
    {
        "entity_id": 12,
        "description": "<e> Zest Air </e> is a former name of Zest Airways, Inc. before rebranding to AirAsia Zest ."
    },
    {
        "entity_id": 13,
        "description": "<e> A year </e> after the strategic alliance with AirAsia , Zest Airways, Inc.  was rebranded as AirAsia Zest."
    },
    {
        "entity_id": 14,
        "description": "<e> AirAsia </e> is an airline that formed a strategic alliance with Zest Air , leading to the rebranding as  AirAsia Zest ."
    },
    {
        "entity_id": 15,
        "description": "<e> AirAsia Philippines </e> is the airline into which Zest Airways, Inc. was merged in January 2016 ."
    },
    {
        "entity_id": 16,
        "description": "<e> January 2016 </e> is when Zest Airways, Inc. was merged into AirAsia Philippines ."
    }
]


```

Real input data:
```json
{real_data}
```

output: