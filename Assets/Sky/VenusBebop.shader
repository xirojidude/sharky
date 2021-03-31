
Shader "Skybox/VenusBebop"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }

// Venus Bebop - Result of an improvised live coding session on Twitch
// Vaguely inspired by the floating islands on Venus in Cowboy Bebop
// LIVE SHADER CODING, SHADER SHOWDOWN STYLE, EVERY TUESDAYS 20:00 Uk time: 
// https://www.twitch.tv/evvvvil_

float2 z,v,e=float2(.0035,-.0035);float t,tt,b,bb,g,gg,tnoi; float3 np,bp,op,pp,po,no,al,ld;
float bo(float3 p,float3 r){ p = abs(p) - r; return max(max(p.x,p.y),p.z);}
float2x2 r2(float r){return float2x2(cos(r),sin(r),-sin(r),cos(r));}
float smin( float d1, float d2, float k ){  float h = max(k-abs(d1-d2),0.0);return min(d1,d2)-h*h*0.25/k; }
float smax( float d1, float d2, float k ){  float h = max(k-abs(-d1-d2),0.0);return max(-d1,d2)+h*h*0.25/k; }
float4 texNoise(float2 uv){ float f = 0.; f+=tex2D(_MainTex, uv*.125).r*.5; f+=tex2D(_MainTex,uv*.25).r*.25; //MERCURTY SDF LIBRARY IS HERE OFF COURSE: http://mercury.sexy/hg_sdf/
                       f+=tex2D(_MainTex,uv*.5).r*.125; f+=tex2D(_MainTex,uv*1.).r*.125; f=pow(f,1.2);return float4(f*.45+.05,f*.45+.05,f*.45+.05,f*.45+.05);}
float cy(float3 p,float3 r){ return max(abs(length(p.xz)-r.x)-r.y,abs(p.y)-r.z);}
float2 tower( float3 p,float flying)
{
    float2 h,t=float2(cy(p,float3(.8,.2,1.2)),5);//BLUE CYL CUT
    t.x=max(t.x,-(abs(abs(abs(p.y)-.4)-.2)-.1));
    t.x=max(t.x,-(abs(p.z)-.2)); 
    h=float2(cy(p,float3(.5,.1,1.4)),6); //WHITE OUTTER
    h.x=max(h.x,-(abs(p.z)-.4));
    h.x=smin(h.x,length(abs(p)-float3(0,1.5,1))-.4,.5);
    h.x=smin(h.x,.6*length(abs(p.xz)-float2(0,1))-.1+abs(p.y)*.02,.5);  
    h.x=min(h.x,bo(p+float3(.7,1,-2.2),float3(2,4,1))); //FLOOR WHITE
    t=t.x<h.x?t:h;    
    if(flying>0.) {
        h=float2(bo(abs(p+float3(.7,1,-2.))-float3(1.6,1.-sin(tt)*2.,0),float3(p.z*.1,p.z*.1,10)),6); //BLACK CUBES FLYING  
        gg+=0.1/(0.1+h.x*h.x*40.);
        t=t.x<h.x?t:h;
    }  
    h=float2(cy(p,float3(.8,.1,1.4)),3);//BLACK CYL  
    h.x=min(h.x,bo(abs(p+float3(.7,1,-2.))-float3(2.,2,0),float3(.1,2,1))); //BLACK EDGES FLOOR 
    h.x=max(h.x,-(abs(p.z)-.3));
    t=t.x<h.x?t:h;
    p.xy=mul(p.xy,r2(-sin(p.y*.2)*.5));  
    t=t.x<h.x?t:h;  
    return t;
}
float4 c=float4(0,5,5,.2);
float2 mp( float3 p)
{
    op=p;
    p.x+=sin(op.z*.1+tt*.2)*2.;
    p.z=((p.z+tt*2.)%20.)-10.;
    float3 tp=p;
    float disp=sin(p.x*.4)+cos(p.z*.2)*.5;
    float2 _uv=float2(op.x,dot(op.yz+float2(0,tt*2.),float2(.5,.5)));
    tnoi=texNoise(_uv*.05).r;  
    float3 towp=2.1*(p-float3(0,1.1,-1.));
    for(int i=0;i<3;i++){
        towp.xz=abs(towp.xz)-float2(1.5,1);
        towp.xz=mul(towp.xz,r2(.55));
    }   
    towp.xz-=1.;
    float2 h,t=tower(towp,0.);t.x/=2.5;  //FIRST ROUND OF TOWER KIFS
    bp=towp; bp.xy=mul(bp.xy,r2(-.3-sin(p.y*.2+tt+op.z*.1)*.4));
    h=float2(.7*(length(bp.xz+float2(-0,2.))-.1+p.y*.05)*.5,6);//LAZER
    h.x=max(h.x,-p.y);
    g+=0.1/(0.1+h.x*h.x*40.);   
    t=t.x<h.x?t:h;   
    towp.yz=mul(towp.yz,r2(.785*2.)); 
    towp.xy-=float2(1.,0.1);
    h=tower(towp,1.);h.x/=2.5;  //SECOND ROUND OF TOWER KIFS
    t=t.x<h.x?t:h;
    h=float2(length(tp)-7.+sin(p.z*.3)*1.5+tnoi*2.5,7);//TERRAIN  STARTS WITH SPHERE
    h.x=smax(-p.y-.2+tnoi*2.,h.x,1.5);  //SMOOTH CUT SPHERE WITH PLANE
    h.x=smax(length(tp-float3(0,0,-1))-1.5+tnoi,h.x,1.5);   //SMOOTH SUBSTRACT SPHERE IN MIDDLE OF ISLAND
    h.x=smin(h.x,p.y+7.+sin(op.x*.15+1.5)+sin(op.z*.3+tt*.6+3.14)*1.5+tnoi*3.,5.); //SMOOTH MIN ADD TERRAIN AT BOTTOM
    h.x=smin(length(abs(tp.xz-float2(2.-tnoi,-1))-4.+disp)-2.+tnoi+(abs(p.y)+5.+tnoi*2.)*0.2,h.x,1.5);//SMOOTH MIN ADD 4 SPIKEY MOUTAINS ON SIDE OF ISLAND
    h.x=smin(length(abs(op.xy+float2(0,1.5))-float2(2.5+sin(op.z*.2+tt*.4)*1.,0.))-1.3+tnoi*2.+sin(op.z*10.+tt*20.)*.03+sin(op.z*.5+tt*2.)*.4,h.x,1.5);  //SMOOTH MIN ADD 2 INFINITE Z CYLINDERS
    h.x*=0.6; pp=tp;
    t=t.x<h.x?t:h;   
    h=float2(length(cos(p*.5+towp*.4+float3(0,0,tt*2.))),6); //PARTICLES
    h.x=max(h.x,length(p.xz)-7.);
    g+=0.1/(0.1+h.x*h.x*200.);
    t=t.x<h.x?t:h;
    return t;
}
float2 tr( float3 ro,float3 rd )
{
    float2 h,t=float2(0.1,.1);
    [loop]
    for(int i=0;i<128;i++){
        h=mp(ro+rd*t.x);
        if(h.x<.0001||t.x>100.) break;
        t.x+=h.x;t.y=h.y;
    }  
    if(t.x>100.) t.y=0.;
    return t;
}
#define a(d) clamp(mp(po+no*d).x/d,0.,1.)
#define s(d) smoothstep(0.,1.,mp(po+ld*d).x/d)

         fixed4 frag (v2f v2) : SV_Target
            {
                float2 fragCoord = v2.vertex;

                float3 viewDirection = normalize(v2.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v2.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

//    float2 uv=(fragCoord.xy/iResolution.xy-0.5)/float2(iResolution.y/iResolution.x,1);
    tt=((16.+_Time.y)%62.82);
//    float3 ro=lerp(float3(1),float3(.3,1.5,1),ceil(sin(tt*.2)))*float3(cos(tt*.4)*9.,3.-cos(tt*.4)*1.5,-15.),
    float3 cw=normalize(float3(0,-6.+sin(tt*.4)*5.,0)-ro),
    cu=normalize(cross(cw,float3(0,1,0))),
    cv=normalize(cross(cu,cw));
//    rd=float3x3(cu,cv,cw)*normalize(float3(uv,.6))
    float3 co,fo;
    float a = atan2(rd.z,rd.x);
    v=float2( abs(a),rd.y ); //-tt*.02);
    co=fo=float3(.13,.1,.1)-length(fragCoord.xy)*.15-rd.y*.15+texNoise(v*.4).r*.3,
    ld=normalize(float3(-.2,.3,-.4));
    z=tr(ro,rd);t=z.x;
    if(z.y>0.){   
        po=ro+rd*t;
        no=normalize(e.xyy*mp(po+e.xyy).x+e.yyx*mp(po+e.yyx).x+e.yxy*mp(po+e.yxy).x+e.xxx*mp(po+e.xxx).x);
        al=lerp(float3(.1,.2,0.7),float3(.1,.6,0.8),.5+.5*sin(pp.y*2.+1.));
        float sp=pow(max(dot(reflect(-ld,no),-rd),0.),40.);
        if(z.y<5.) al=float3(0,0,0);
        if(z.y>5.) al=float3(1,1,1);
        if(z.y>6.) sp=0.,al=lerp(float3(1,1,1),float3(.2,.4,.5),.5+.5*sin(min(-pp.y*.7+.7,2.)+.2+pp.x*.1))-tnoi*1.8;
        float dif=max(0.,dot(no,ld)),
        fr=pow(1.+dot(no,rd),4.);    
        co=lerp(sp+al*(a(.05)*a(.1)+.2)*(dif+s(3.)),fo,min(fr,.5));
        co=lerp(fo,co,exp(-.00002*t*t*t));
    }
    fragColor = float4(pow(co+g*.2*float3(.1,.2,.7)+gg*.2*float3(.7,.2,.1),float3(.55,.55,.55)),1);


                return fragColor;
            }

            ENDCG
        }
    }
}



