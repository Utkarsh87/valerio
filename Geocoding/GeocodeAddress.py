import geocoder

def geocoding(address):
	g = geocoder.osm(address)
	latitude = g.osm['y']
	longitude = g.osm['x']
	return latitude, longitude