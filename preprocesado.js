print("Aplicando operaciones de preprocesado, esto puede tardar un rato dependiendo de la cantidad de tweets...")

// Eliminar los campos generales que no nos interesan:

db.tweets.update({}, { $unset: {id_str: true, source: true, truncated: true, display_text_range: true, in_reply_to_status_id: true, in_reply_to_status_id_str: true, in_reply_to_user_id : true, in_reply_to_user_id_str: true, in_reply_to_screen_name : true, is_quote_status: true, quote_count : true, reply_count : true, retweet_count: true, favorite_count : true, filter_level: true, timestamp_ms: true, possibly_sensitive: true, contributors: true, quoted_status_id: true, quoted_status_id_str: true, favorited: true, retweeted: true, lang: true, geo: true, coordinates: true, place: true} }, {multi:true});

print("Estamos trabajando en ello... paso 1/3")

// Marcar RT:

db.tweets.update({retweeted_status: { $exists: true}}, { $set: {RT: true} }, {multi:true});

// Marcar número de RT de un tweet original: ---> índices o no funciona

db.tweets.createIndex({id: 1}, {"name": "id_tweet"});

db.tweets.aggregate([
	{ "$match": { "RT": true }},
    { "$group": {
        "_id": "$retweeted_status.id",
        "count": { "$sum": 1 }
    }},
    {"$out": "num_rts"}
], {allowDiskUse: true});

db.num_rts.find().forEach(function(doc) {
    db.tweets.updateOne({"id":doc._id},{$set: {num_RT : doc.count}});  
});

db.tweets.update({RT: true}, {$unset: {retweeted_status: true}}, {multi: true});
db.tweets.update({RT: {$exists: false}}, {$set: {RT: false} }, {multi:true});
db.tweets.update({RT: false, num_RT: {$exists: false}}, {$set: {num_RT : 0}}, {multi: true});

print("Estamos trabajando en ello... paso 2/3")

// Eliminar datos de usuario que no nos interesan:
db.tweets.update({}, {$unset: {'user.id_str': true, 'user.name': true, 'user.description': true, 'user.translator_type': true, 'user.protected': true, 'user.friends_count': true, 'user.listed_count': true, 'user.favourites_count': true, 'user.created_at': true, 'user.utc_offset': true, 'user.time_zone': true, 'user.geo_enabled': true, 'user.lang': true, 'user.contributors_enabled': true, 'user.is_translator': true, 'user.url': true, 'user.statuses_count': true, 'user.profile_background_color': true, 'user.profile_background_image_url': true, 'user.profile_background_image_url_https': true, 'user.profile_background_tile': true, 'user.profile_link_color': true, 'user.profile_sidebar_border_color': true, 'user.profile_sidebar_fill_color': true, 'user.profile_text_color': true, 'user.profile_use_background_image': true, 'user.profile_image_url': true, 'user.profile_image_url_https': true, 'user.profile_banner_url': true, 'user.default_profile': true, 'user.default_profile_image': true,  'user.following': true, 'user.follow_request_sent': true, 'user.notifications': true, 'user.location': true, 'user.followers_count': true }}, {multi: true});

// Extraer menciones, hashtags y urls a array aparte
db.tweets.update({}, {$unset: {urls: true, mentions: true, hashtags: true}}, {multi: true});

db.tweets.update({}, {$set: {urls: [], mentions: [], hashtags: []}}, {multi: true});

db.tweets.find({}).forEach(function (doc){
		doc.entities.hashtags.forEach( function (ht){
				doc.hashtags.push(ht.text) 
			
		});
		doc.entities.urls.forEach( function (url){
				doc.urls.push(url.url) 
			
		});
		doc.entities.user_mentions.forEach( function (mention){
				doc.mentions.push(mention.screen_name) 
			
		});
		if (doc.urls.length == 0){
			delete doc.urls;
		}
		if (doc.hashtags.length == 0){
			delete doc.hashtags;
		}
		if (doc.mentions.length == 0){
			delete doc.mentions;
		}
		db.tweets.save(doc);
	});

print("Estamos trabajando en ello... paso 3/3")

// Eliminar campo entities, ya usado
db.tweets.update({}, {$unset: {entities: true, extended_entities: true}}, {multi: true});

db.tweets.find({RT: false}).forEach(function(doc){
     text = doc["text"];
     textWithoutUrls = text.replace(/\bhttps\S+/ig, "");
     textWithoutUrlsAndMentions = textWithoutUrls.replace(/\@\S+/ig, "");
     textWithoutUrlsEmojisAndMentions = textWithoutUrlsAndMentions.replace(/([\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2694-\u2697]|\uD83E[\uDD10-\uDD5D])/g, '').trim();
     doc["text2"] = textWithoutUrls;
     doc["text3"] = textWithoutUrlsEmojisAndMentions;

     if (doc["text3"] == ""){
     	doc["voidtext"] = true;
     }
    doc.string_created_at = doc.created_at;
	doc.created_at = new Date(doc.created_at); 
    db.tweets.update({_id:doc["_id"]},doc); 
  });

// Encontrar tweets repetidos que se importaron mal por algo y eliminar duplicados:
db.tweets.aggregate([
    { "$group": {
        "_id": "$id",
        "dups": { "$push": "$_id" },
        "count": { "$sum": 1 }
    }},
    { "$match": { "count": { "$gt": 1 } }}
], {allowDiskUse: true}).forEach(function(doc) {
    doc.dups.shift();
    db.tweets.remove({ "_id": {"$in": doc.dups }});
});


// Crear índice por created_at porque sino no cabe en RAM al traerse los tweets
db.tweets.createIndex({created_at: 1}, {"name": "orden_fecha"});

print("Preprocesado terminado!!")
