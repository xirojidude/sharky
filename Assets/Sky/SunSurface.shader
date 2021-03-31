
Shader "Skybox/SunSurface"
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


// Based on Shanes' Fiery Spikeball https://www.shadertoy.com/view/4lBXzy (I think that his implementation is more understandable than the original :) ) 
// Relief come from Siggraph workshop by Beautypi/2015 https://www.shadertoy.com/view/MtsSRf
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0

//#define ULTRAVIOLET
#define DITHERING

#define pi 3.14159265
#define R(p, a) p=cos(a)*p+sin(a)*float2(p.y, -p.x)

// IQ's noise
float pn( in float3 p )
{
    float3 ip = floor(p);
    p = frac(p);
    p *= p*(3.0-2.0*p);
    float2 uv = (ip.xy+float2(37.0,17.0)*ip.z) + p.xy;
    uv = tex2D( _MainTex, (uv+ 0.5)/256.0 ).yx;    //tex2DLod( _MainTex, (uv+ 0.5)/256.0, 0.0 ).yx;
    return lerp( uv.x, uv.y, p.z );
}

// FBM
float fpn(float3 p) {
    return pn(p*.06125)*.57 + pn(p*.125)*.28 + pn(p*.25)*.15;
}

float rand(float2 co){// implementation found at: lumina.sourceforge.net/Tutorials/Noise.html
    return frac(sin(dot(co*0.123,float2(12.9898,78.233))) * 43758.5453);
}

float cosNoise( in float2 p )
{
    return 0.5*( sin(p.x) + sin(p.y) );
}

const float2x2 m2 = float2x2(1.6,-1.2,
                     1.2, 1.6);

float sdTorus( float3 p, float2 t )
{
  return length( float2(length(p.xz)-t.x*1.2,p.y) )-t.y;
}

float smin( float a, float b, float k )
{
    float h = clamp( 0.5 + 0.5*(b-a)/k, 0.0, 1.0 );
    return lerp( b, a, h ) - k*h*(1.0-h);
}

float SunSurface( in float3 pos )
{
    float h = 0.0;
    float2 q = pos.xz*0.5;
    
    float s = 0.5;
    
    float d2 = 0.0;
    
    for( int i=0; i<6; i++ )
    {
        h += s*cosNoise( q ); 
        q = mul(m2,q*0.85); 
        q += float2(2.41,8.13);
        s *= 0.48 + 0.2*h;
    }
    h *= 2.0;
    
    float d1 = pos.y - h;
   
    // rings
    float3 r1 = ((2.3+pos+1.0)%10.0)-5.0;
    r1.y = pos.y-0.1 - 0.7*h + 0.5*sin( 3.0*_Time.y+pos.x + 3.0*pos.z);
    float c = cos(pos.x); float s1 = 1.0;//sin(pos.x);
    r1.xz=c*r1.xz+s1*float2(r1.z, -r1.x);
    d2 = sdTorus( r1.xzy, float2(clamp(abs(pos.x/pos.z),0.7,2.5), 0.20) );

    
    return smin( d1, d2, 1.0 );
}

float map(float3 p) {
   p.z += 1.;
   R(p.yz, -25.5);// -1.0+iMouse.y*0.003);
   R(p.xz, 1*0.008*pi+_Time.y*0.1);
   return SunSurface(p) +  fpn(p*50.+_Time.y*25.) * 0.45;
}

// See "Combustible Voronoi"
// https://www.shadertoy.com/view/4tlSzl
float3 firePalette(float i){

    float T = 1400. + 1300.*i; // Temperature range (in Kelvin).
    float3 L = float3(7.4, 5.6, 4.4); // Red, green, blue wavelengths (in hundreds of nanometers).
    L = pow(L,float3(5.0,5,5)) * (exp(1.43876719683e5/(T*L))-1.0);
    return 1.0-exp(-5e8/L); // Exposure level. Set to "50." For "70," change the "5" to a "7," etc.
}



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

   // p: position on the ray
   // rd: direction of the ray
//   float3 rd = normalize(float3((gl_FragCoord.xy-0.5*iResolution.xy)/iResolution.y, 1.));
//   float3 ro = float3(0., 0., -22.);
   ro = ro = float3(0., 0., -22.);
   
   // ld, td: local, total density 
   // w: weighting factor
   float ld=0., td=0., w=0.;

   // t: length of the ray
   // d: distance function
   float d=1., t=1.;
   
   // Distance threshold.
   const float h = .1;
    
   // total color
   float3 tc = float3(0.,0,0);
   
   #ifdef DITHERING
   float2 pos = ( fragCoord.xy ); /// iResolution.xy );
   float2 seed = pos + frac(_Time.y);
   //t=(1.+0.2*rand(seed));
   #endif
    
   // rm loop
   for (int i=0; i<56; i++) {

      // Loop break conditions. Seems to work, but let me
      // know if I've overlooked something.
      if(td>(1.-1./80.) || d<0.001*t || t>40.)break;
       
      // evaluate distance function
      d = map(ro+t*rd); 
       
      // fix some holes deep inside
      //d=max(d,-.3);
      
      // check whether we are close enough (step)
      // compute local density and weighting factor 
      //const float h = .1;
      ld = (h - d) * step(d, h);
      w = (1. - td) * ld;   
     
      // accumulate color and density
      tc += w*w + 1./50.;  // Different weight distribution.
      td += w + 1./200.;

      // dithering implementation come from Eiffies' https://www.shadertoy.com/view/MsBGRh
      #ifdef DITHERING  
      #ifdef ULTRAVIOLET
      // enforce minimum stepsize
      d = max(d, 0.04);
      // add in noise to reduce banding and create fuzz
      d=abs(d)*(1.+0.28*rand(seed*float2(i)));
      #else
      // add in noise to reduce banding and create fuzz
      d=abs(d)*(.8+0.28*rand(seed*float2(i,i)));
      // enforce minimum stepsize
      d = max(d, 0.04);
      #endif 
      #else
      // enforce minimum stepsize
      d = max(d, 0.04);        
      #endif

      // step forward
      t += d*0.5;
      
   }

   // Fire palette.
   tc = firePalette(tc.x);
   
   #ifdef ULTRAVIOLET
   tc *= 1. / exp( ld * 2.82 ) * 1.05;
   #endif
    
   fragColor = float4(tc, 1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}

