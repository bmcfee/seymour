// nothing here yet

$(document).ready(function() {
    $('#myform').submit(function(e) {

        e.preventDefault();
        $('#offset').val(0);
        var offset          = parseInt($('#offset').val());
        var limit           = parseInt($('#limit').val());
        search(offset, limit);

    })
});

function dv(x, v) {

    if ((typeof x) !== undefined) {
        return x;
    }
    return v;
}


$('.previous').click(function() {
    if ($(this).hasClass('disabled')) {
        return;
    }
    var offset          = parseInt($('#offset').val());
    var limit           = parseInt($('#limit').val());

    search(Math.max(offset - limit, 0), limit);
});

$('.next').click(function() {
    if ($(this).hasClass('disabled')) {
        return;
    }
    var track_count     = parseInt($('#track_count').val());
    var offset          = parseInt($('#offset').val());
    var limit           = parseInt($('#limit').val());

    search(Math.min(offset + limit, track_count-1), limit);
});

function search(offset, limit) {


    offset = dv(offset, 0);
    limit  = dv(limit, 18);

    offset = parseInt(offset);
    limit  = parseInt(limit);

    // First update the collection container
    $.ajax({url: '/search/' + $('#query').val() + '/' + offset + '/' + limit,
            dataType: 'json'}).done(function(data) {

            $("#tracklist > *").remove();

            if (data.offset == 0) {
                $('.previous').addClass('disabled');
            } else {
                $('.previous').removeClass('disabled');
            }
            $('#offset').val(data.offset);
            $('#limit').val(data.limit);

            var track_container = $("#tracklist");

            var start = 1 + data.offset;
            var end   = data.offset + data.results.length;

            $("#track-min").text(start);
            $("#track-max").text(end);

            for (var i in data.results) {
                
                var content = $('<a>')
                        .addClass('list-group-item')
                        .addClass('col-md-3')
                        .addClass('track')
                        .attr('href', '/track/'+ data.results[i].track_id)
                        .attr('target', '_blank');


                content.append($('<h4>')
                        .addClass('list-group-item-heading')
                        .text(data.results[i].title));

                var ctext = data.results[i].artist;
                if (data.results[i].album.length > 0) {
                    ctext += ' - ' + data.results[i].album;
                }

                content.append($('<p>')
                        .addClass('list-group-item-text')
                        .addClass('text-muted')
                        .text(ctext));
                
                track_container.append(content);
            }

            $("#track-total").text(data.count);
            $('#track_count').val(data.count);

            if (data.count > offset + data.results.length) {
                $('.next').removeClass('disabled');
            } else {
                $('.next').addClass('disabled');
            }

            if (data.count > 0) {
                $('#results').removeClass('hidden');
                $('#noresults').addClass('hidden');
            } else {
                $('#results').addClass('hidden');
                $('#noresults').removeClass('hidden');
            }
    });

}
