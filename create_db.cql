//
// COFFEE
//
CREATE CONSTRAINT ON (c:Coffee) ASSERT c.id IS UNIQUE;


LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/jacoma/Project_Mod/main/neo_coffee.csv' AS line
WITH $dict.rows as rows UNWIND rows as row
        MERGE (p:Payer {id:row.Id})
          SET p.name=row.NAME,
            p.address=row.ADDRESS,
            p.city=row.CITY,
            p.zip=row.ZIP,
            p.state=row.STATE_HEADQUARTERED


LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/jacoma/scratch/main/neo_coffee.csv' AS line
WITH line, SPLIT(line.varietals, ",") AS var_list
UNWIND var_list AS variety
MATCH (coffee:Coffee {id:toInteger(line.coffee_id)})
CREATE (coffee)-[:HAS_TYPE]->(variety)


LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/jacoma/scratch/main/neo_coffee.csv' AS line

MATCH (coffee:Coffee {id:toInteger(line.coffee_id)}), (roaster:Roaster {id:roaster_id})
CREATE (coffee)-[:ROASTED_BY]->(roaster)




//
// ROASTER
//
CREATE CONSTRAINT ON (r:Roaster) ASSERT r.roaster_id IS UNIQUE;
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/jacoma/scratch/main/neo_roaster.csv' AS line

CREATE (roaster:Roaster {id:TOINT(line.roaster_id), name:line.name, website:line.website})



//
// COUNTRY
//
CREATE CONSTRAINT ON (c:Country) ASSERT c.country_code IS UNIQUE;
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/jacoma/scratch/main/neo_country.csv' AS line

CREATE (country:Country {id:toInteger(line.country_code), name:line.name, country_code_alpha:line.country_code_alpha,latitude:toFloat(line.latitude),longitude:toFloat(line.longitude)})
MERGE (subRegion:subRegion {name:line.subRegion})
MERGE (region:Region {name:line.Region})


CREATE CONSTRAINT ON (r:Region) ASSERT r.name IS UNIQUE;
CREATE CONSTRAINT ON (s:subRegion) ASSERT s.name IS UNIQUE;
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/jacoma/scratch/main/neo_country.csv' AS line

MATCH (country:Country {id:toInteger(line.country_code)}), (region:Region {name:line.region}), (sub:subRegion {name:line.subregion})

CREATE (country)-[:IN_LOCATION]->(sub)
MERGE (sub) -[:IN_LOCATION]->(region)



//
// VARIETALS
//
CREATE CONSTRAINT ON (v:Varietal) ASSERT v.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/jacoma/scratch/main/neo_distinct_variety.csv' AS line

CREATE (var:Varietal {id:toInteger(line.id), name:line.varietal_3})


LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/jacoma/scratch/main/neo_variety.csv' AS line

MATCH (var:Varietal {id:toInteger(line.id))
MATCH (var2:Varietal {name:line.varietal_2})

CREATE (var)-[:IN_CATEGORY]->(var2)