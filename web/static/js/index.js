// nothing here yet

$(document).ready(function() {
//     search_graph($('#seed_id').val(), $('#depth').val());
    // Query for the collections list
    $.ajax({ url: '/collections', dataType: 'json'}).done(update_collections);
});

function dv(x, v) {

    if ((typeof x) !== undefined) {
        return x;
    }
    return v;
}

function update_collections(collections) {
    // Clear the collections list
    $("#collections-menu > *").remove();

    // Populate the collections list
    var menu = $("#collections-menu");

    for (var c in collections) {
        var li = $('<li role="presentation">');
    
        var button = $('<button type="button" class="btn btn-link"></button>');
        button.text(collections[c].name);
        var hidden = $('<input type="hidden">');
        hidden.val(collections[c].collection_id);

        button.append(hidden);

        button.click(function() {
            var hidden = $(this).find('input:hidden')[0];

            get_collection($(hidden).val(), 0, 15);
        });
        li.append(button);
        menu.append(li);
    }

    // Set the default and update the pager
    get_collection(collections[0].collection_id, 0, 15);
}


$('.previous').click(function() {
    if ($(this).hasClass('disabled')) {
        return;
    }
    var offset          = parseInt($('#offset').val());
    var limit           = parseInt($('#limit').val());
    var collection_id   = parseInt($('#collection_id').val());

    get_collection(collection_id, offset - limit, limit);
});

$('.next').click(function() {
    if ($(this).hasClass('disabled')) {
        return;
    }
    var offset          = parseInt($('#offset').val());
    var limit           = parseInt($('#limit').val());
    var collection_id   = parseInt($('#collection_id').val());

    get_collection(collection_id, offset + limit, limit);
});

function get_collection(collection_id, offset, limit) {


    offset = dv(offset, 0);
    limit  = dv(limit, 15);

    offset = parseInt(offset);
    limit  = parseInt(limit);
    collection_id = parseInt(collection_id);

    // First update the collection container
    $.ajax({url: '/tracks/' + collection_id + '/' + offset + '/' + limit,
            dataType: 'json'}).done(function(tracklist) {
            $("#tracklist > *").remove();
            if (offset == 0) {
                $('.previous').addClass('disabled');
            } else {
                $('.previous').removeClass('disabled');
            }
            $('#offset').val(offset);
            $('#limit').val(limit);
            $('#collection_id').val(collection_id);

            var track_container = $("#tracklist");

            var start = 1 + offset;
            var end   = offset + tracklist.length;
            $("#track-min").text(start);
            $("#track-max").text(end);

            for (var i in tracklist) {
                var li = $('<li></li>').addClass('list-group-item').addClass('track');
                
                var content = $('<div></div>');
                content.append($('<h4></h4>').text(tracklist[i].title));
                content.append($('<h5></h5>')
                        .addClass('text-muted')
                        .text(tracklist[i].artist + ' - ' + tracklist[i].album));
                
                var hidden = $('<input type="hidden">').val(tracklist[i].track_id);
                li.append(hidden);
                li.append(content);

                li.click(function() {
                    var hid = $($(this).find('input:hidden')[0]);
//                     $(location).attr('href', '/track/' + hid.val());
                    window.open('/track/' + hid.val());
                });
                track_container.append(li);
            }
            // Update the collection container object
            $.ajax({url: '/collection/' + collection_id, dataType: 'json'}).done(
                function(data) {
                    $("#track-total").text(data.track_count);
                    $("#collection-name").text(data.name);
                    $('#track_count').val(data.track_count);

                    if (data.track_count > offset + tracklist.length) {
                        $('.next').removeClass('disabled');
                    } else {
                        $('.next').addClass('disabled');
                    }
                }
            );
    });

}
