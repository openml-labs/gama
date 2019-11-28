This is a Dash app to configure and start GAMA through UI.
While this work will be merged with the Visualization app, for now they are separate.
The distinction is that the Visualization app was built to compare multiple runs to each other,
while this app is built to use GAMA and monitor live ML pipeline search.

---

A word on the usage of Dash. This (and the for now separate Visualization app) is my first Dash app.
I was (and still am) not sure what the right tools for developing this GUI are.
Dash looked good for prototyping the tool.

I'm sure there's lots to be improved. But in particular know that the usage of local python objects for
storing data is wrong from a Dash perspective - as this means state is stored in the server and not the client.
At this point I have no plan to support multiple clients, so I'm okay with this.
Another point of pain is that I'm currently running GAMA as a subprocess and communication
with the GUI is facilitated through reading and parsing the gama log file.
I'd very much like for this to be replaced by communication within Python (e.g. events).
