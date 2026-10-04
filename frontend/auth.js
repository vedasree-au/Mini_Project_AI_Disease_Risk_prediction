const API_BASE = localStorage.getItem("apiBaseUrl") || "http://127.0.0.1:8000";
function authMessage(id,text,type="error"){const el=document.getElementById(id);el.className=`auth-message ${type}`;el.textContent=text;el.style.display="block";}
async function registerUser(){
 const payload={username:document.getElementById("register-username").value.trim(),full_name:document.getElementById("register-fullname").value.trim(),email:document.getElementById("register-email").value.trim(),phone:document.getElementById("register-phone").value.trim(),password:document.getElementById("register-password").value};
 if(!payload.username||!payload.email||!payload.password)return authMessage("register-result","Please complete username, email and password.");
 if(payload.password.length<6)return authMessage("register-result","Password must contain at least 6 characters.");
 try{const r=await fetch(`${API_BASE}/register`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.detail||"Registration failed");localStorage.setItem("authToken",d.token);localStorage.setItem("loggedInUser",JSON.stringify(d.user||{username:payload.username,email:payload.email,full_name:payload.full_name,phone:payload.phone}));authMessage("register-result","Account created. Opening your dashboard…","success");setTimeout(()=>location.href="index.html",700);}catch(e){authMessage("register-result",e.message);}}
async function loginUser(){
 const email=document.getElementById("login-email").value.trim(),password=document.getElementById("login-password").value;
 if(!email||!password)return authMessage("login-result","Please enter your email and password.");
 try{const r=await fetch(`${API_BASE}/login`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({email,password})});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.detail||"Incorrect email or password.");localStorage.setItem("authToken",d.token);localStorage.setItem("loggedInUser",JSON.stringify(d.user));authMessage("login-result","Signed in successfully. Loading dashboard…","success");setTimeout(()=>location.href="index.html",500);}catch(e){authMessage("login-result",e.message);}}
if(location.pathname.endsWith("index.html")&& !localStorage.getItem("authToken")) location.href="login.html";
