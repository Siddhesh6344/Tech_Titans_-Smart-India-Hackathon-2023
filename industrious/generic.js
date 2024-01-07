const tbody = document.getElementById('tbody');
const url = 'http://localhost:4000/results';
setInterval(() => {
    fetch(url)
    .then((response) => {
      return response.json();
    })
    .then((data) => {
      let authors = data;
      tbody.innerHTML="";

      authors.map(function(author) {
        let tr = document.createElement('tr');
        tr.innerHTML=`<td>${author.start_time}</td>
        <td>${author.end_time}</td>
        <td>${author.labels}</td>
        <td>${author.total_time_min}</td>`
        
        tbody.appendChild(tr);
      });
    })
    .catch(function(error) {
      console.log(error);
    });
},1000);