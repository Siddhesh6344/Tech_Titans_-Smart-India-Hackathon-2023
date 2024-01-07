const tbody = document.getElementById('tbody');
const url = 'http://localhost:4000';
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
        tr.innerHTML=`<td>${author.img_read}</td>
        <td>${author.img_label}</td>
        <td>${author.label}</td>`
        
        tbody.appendChild(tr);
      });
    })
    .catch(function(error) {
      console.log(error);
    });
},1000);